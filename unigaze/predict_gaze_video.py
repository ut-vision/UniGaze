
import os, argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import cv2
import torch

from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw, angular_error
from gazelib.gaze.normalize import estimateHeadPose, normalize
from gazelib.label_transform import get_face_center_by_nose

from utils import instantiate_from_cfg
from datasets.helper.image_transform import wrap_transforms
import face_alignment


def draw_gaze(image_in, pitchyaw, thickness=8, color=(0, 0, 255)):
	"""Draws a more 3D-like gaze vector on the image."""
	image_out = image_in.copy()
	(h, w) = image_in.shape[:2]
	length = w / 2.0
	pos = (int(h / 2.0), int(w / 2.0))
	
	# Convert grayscale image to BGR if needed
	if len(image_out.shape) == 2 or image_out.shape[2] == 1:
		image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
	
	# Calculate the gaze direction end point
	dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
	dy = -length * np.sin(pitchyaw[0])
	end_point = (int(pos[0] + dx), int(pos[1] + dy))
	
	
	# Add a shadow effect for a 3D look
	shadow_offset = 2
	shadow_color = (40, 40, 40)  # Dark grey for shadow
	shadow_end = (end_point[0] + shadow_offset, end_point[1] + shadow_offset)
	cv2.arrowedLine(image_out, (pos[0] + shadow_offset, pos[1] + shadow_offset), shadow_end, shadow_color, thickness + 2, cv2.LINE_AA, tipLength=0.3)

	# Draw the main arrow with gradient layers to simulate depth
	thickness_values = [4,3,2,1]
	num_layers = len(thickness_values)
	for i in range(num_layers):
		alpha = i / num_layers
		layer_color = tuple(int((1 - alpha) * color[j] + alpha * 255) for j in range(3))  # Blend color towards white
		cv2.arrowedLine(
			image_out, pos, end_point, layer_color, thickness_values[i],
			cv2.LINE_AA, tipLength=0.3
		)

	return image_out




def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
	pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1,2)).reshape(3,1) # get 3d gaze direction as a vector

	pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3,1)) # apply inverse transformation to convert it back to camera coord system
	pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor) # vector normalization
	
	pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1,3)) # convert to yaw and pitch
	return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor





def get_parser(**parser_kwargs):
	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		elif v.lower() in ("no", "false", "f", "n", "0"):
			return False
		else:
			raise argparse.ArgumentTypeError("Boolean value expected.")

	parser = argparse.ArgumentParser(**parser_kwargs)


	parser.add_argument(
		"-i", "--input_dir", help="the path to the input: should be a video", 
	)
	parser.add_argument(
		"-out", "--output_dir", help="the path to save the drawn images", 
	)
	parser.add_argument(
		"-m", "--model_cfg_path", help="the path to the model config file",
	)
	
	parser.add_argument(
		"--model_name", help="the model name tag when loading directly from unigaze package", default=None, type=str,
	)
	parser.add_argument(
		"--ckpt_resume", help="resume from checkpoint", default=None, type=str,
	)

	parser.add_argument(
		"--write_normalized_image", help="whether to write the normalized images", default=False, type=str2bool,
	)

	parser.add_argument(
		"--write_image", help="whether to write the images", default=False, type=str2bool,
	)
	
	return parser




def set_dummy_camera_model(image=None):
	h, w = image.shape[:2]
	focal_length = w*4
	center = (w//2, h//2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
		[0, focal_length, center[1]],
		[0, 0, 1]], dtype = "double"
	)
	camera_distortion = np.zeros((1, 5)) # Assuming no lens distortion
	return np.array(camera_matrix), np.array(camera_distortion)

def load_checkpoint(model, ckpt_key, ckpt_path):
	"""
	Load the copy of a model.
	"""
	assert os.path.isfile(ckpt_path)
	weights = torch.load(ckpt_path, map_location='cpu')
	print('loaded ckpt from : ', ckpt_path)

	# If was stored using DataParallel but being read on 1 GPU
	model_state = weights[ckpt_key]
	if next(iter(model_state.keys())).startswith('module.'):
		print(' convert the DataParallel state to normal state')
		model_state = dict([(k[7:], v) for k, v in model_state.items()])

	model.load_state_dict(model_state, strict=True)
	print(f'loaded {ckpt_key}')
	del weights


arrow_colors = [
		(47, 255, 173), ## green yellow
	]


if __name__ == "__main__":
	args, unknown = get_parser().parse_known_args()

	write_normalized_image = args.write_normalized_image
	write_image = args.write_image

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if args.model_name is not None:
		import unigaze
		model = unigaze.load(args.model_name, device=device)
		model.eval()
	else:
		pretrained_model_cfg=OmegaConf.load(args.model_cfg_path)['net_config']
		pretrained_model_cfg.params.custom_pretrained_path = None  ## since we load the gaze trained checkpoint, this MAE pre-trained checkpoint is not needed
		model = instantiate_from_cfg( pretrained_model_cfg )
		load_checkpoint(model, 'model_state', args.ckpt_resume)
		model.eval()
		model.to(device)

	image_torch_transform = wrap_transforms('basic_imagenet', image_size=224)
	focal_norm = 960 # focal length of normalized camera
	distance_norm = 600  # normalized distance between eye and camera
	roi_size = (224, 224)  # size of cropped eye image
	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
	except:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

	resize_factor = 0.5

	video_paths = sorted(glob(args.input_dir + '/*.mp4')) 
	print("video_paths: ", video_paths)
	for input_path in video_paths:
		input_name = os.path.basename(input_path).split('.')[0]

		## use cv2 to read the video file
		cap = cv2.VideoCapture(input_path)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


		log_info = '============================ Video info ===================================\n'
		log_info += f'The input video : {input_path }\n'
		log_info += f'resolution: {width} x {height},   fps: {fps}\n'
		
		if args.output_dir is None:
			output_dir = os.path.join( os.path.dirname(input_path), f'output')
		else:
			output_dir = args.output_dir
		os.makedirs(output_dir, exist_ok=True)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		filename = input_name + "_pred.mp4"
		out = cv2.VideoWriter(os.path.join(output_dir, filename), fourcc, int(fps), (width, height))
		log_info += f'The recorded video will be saved in: {os.path.join(output_dir, filename)}\n'

		image_output_folder = os.path.join(output_dir, input_name)
		os.makedirs(image_output_folder, exist_ok=True)
		log_info += f'The images will be saved in: {image_output_folder}\n'
		log_info += '\n'
		print(log_info)

		save_freq = 30

		frame_idx = 0
		pbar = tqdm(total=num_frames)
		while(True):
			pbar.update(1)
			ret, image_original = cap.read()
			if not ret:
				break

			if resize_factor >= 1:
				image_resize = image_original.copy()
			else:
				image_resize = cv2.resize(image_original, dsize=None, fx=resize_factor, fy=resize_factor, interpolation = cv2.INTER_AREA)
			
			image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
			preds = fa.get_landmarks(image_resize)


			if preds is not None:
				landmarks_record = {}
				vector_start_end_point_list = {}
				bbox_record = {}
				
				for idx in range(len(preds)):
					color = arrow_colors[idx % len(arrow_colors)]
					landmarks_in_original = preds[idx] # array (68,2)
					landmarks_in_original /= resize_factor

					x_min = int(landmarks_in_original[:, 0].min())
					x_max = int(landmarks_in_original[:, 0].max())
					y_min = int(landmarks_in_original[:, 1].min())
					y_max = int(landmarks_in_original[:, 1].max())
					## each pred is a detected face
					

					## scale the bounding box by scale factor

					scale_factor =1.2
					bbox_width = x_max - x_min
					bbox_height = y_max - y_min
					bbox_center = ( (x_min + x_max) // 2, (y_min + y_max) // 2 )
					x_min_draw = max(0, bbox_center[0] - int(bbox_width * scale_factor // 2))
					x_max_draw = min(image_original.shape[1], bbox_center[0] + int(bbox_width * scale_factor // 2))
					y_min_draw = max(0, bbox_center[1] - int(bbox_height * scale_factor // 2))
					y_max_draw = min(image_original.shape[0], bbox_center[1] + int(bbox_height * scale_factor // 2))
					bbox_record[idx] = (x_min_draw, y_min_draw, x_max_draw, y_max_draw)


					scale_factor = 2.0
					bbox_width = x_max - x_min
					bbox_height = y_max - y_min
					bbox_center = ( (x_min + x_max) // 2, (y_min + y_max) // 2 )
					x_min = max(0, bbox_center[0] - int(bbox_width * scale_factor // 2))
					x_max = min(image_original.shape[1], bbox_center[0] + int(bbox_width * scale_factor // 2))
					y_min = max(0, bbox_center[1] - int(bbox_height * scale_factor // 2))
					y_max = min(image_original.shape[0], bbox_center[1] + int(bbox_height * scale_factor // 2))
					

					image = image_original[y_min:y_max, x_min:x_max]
					landmarks = landmarks_in_original - np.array([x_min, y_min])

					

					################# normalization #################
					camera_matrix, camera_distortion = set_dummy_camera_model(image=image)
					face_model_load = np.loadtxt( 'data/face_model.txt')
					face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
					facePts = face_model.reshape(6, 1, 3)

					landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
					landmarks_sub_paint = landmarks_sub
					landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
					landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
					hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
					hR = cv2.Rodrigues(hr)[0]  # rotation matrix
					face_center_camera_cord, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)
					
					# -------------------------------------------- normalize image --------------------------------------------
					img_normalized, R, hR_norm, gaze_normalized, landmarks_normalized, _ = normalize(image, landmarks, focal_norm, distance_norm, roi_size, face_center_camera_cord, hr, ht, camera_matrix, gc=None)
					
					hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
					if np.linalg.norm(hr_norm) > 80 * np.pi / 180 :
						continue

					input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
					input_var = image_torch_transform(input_var)
					input_var = torch.autograd.Variable(input_var.float().to(device))
					input_var = input_var.unsqueeze(0)
					ret = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation
					
					pred_gaze = ret["pred_gaze"][0]
					pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
					img_normalized = draw_gaze(img_normalized, pred_gaze_np, thickness=5, color=color)

					if write_normalized_image or frame_idx % save_freq == 0:
						# print( f"frame: {frame_idx}, idx: {idx}")
						cv2.imwrite(os.path.join(image_output_folder, input_name + f'_{frame_idx}_{idx}_normalize.jpg'), img_normalized)
					

					R_inv = np.linalg.inv(R)
					pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor = denormalize_predicted_gaze(pred_gaze_np, R_inv)
					## project the 3D Gaze back to 2D image
					vec_length = pred_gaze_cancel_nor * -112 * 1.5
					gazeRay = np.concatenate((face_center_camera_cord.reshape(1,3), (face_center_camera_cord + vec_length).reshape(1,3)), axis=0)
					result = cv2.projectPoints( gazeRay, 
											np.array([0,0,0]).reshape(3,1).astype(float),
											np.array([0,0,0]).reshape(3,1).astype(float), 
											camera_matrix, camera_distortion )
					result = result[0].reshape(2,2)
					result += np.array([x_min, y_min])
					
					vector_start_point =  (int(result[0][0]), int(result[0][1]))
					vector_end_point = (int(result[1][0]), int(result[1][1]))

					vector_start_end_point_list[idx] = (vector_start_point, vector_end_point)
					landmarks_record[idx] = landmarks_in_original
					# bbox_record[idx] = (x_min, y_min, x_max, y_max)



				for idx in list(landmarks_record.keys()):
					
					x_min, y_min, x_max, y_max = bbox_record[idx] ## this is just for draw

					color = arrow_colors[idx % len(arrow_colors)]

					cv2.rectangle(image_original, (x_min, y_min), (x_max, y_max), (0, 0, 240), 2)
					#### draw gaze
					vector_start_point, vector_end_point = vector_start_end_point_list[idx]
					# image_original = cv2.arrowedLine(image_original, vector_start_point, vector_end_point, color, thickness=3)
					# Add a shadow effect for a 3D look
					shadow_offset = 2
					shadow_color = (40, 40, 40)  # Dark grey for shadow
					shadow_end = (vector_end_point[0] + shadow_offset, vector_end_point[1] + shadow_offset)
					cv2.arrowedLine(image_original, (vector_start_point[0] + shadow_offset, vector_start_point[1] + shadow_offset), shadow_end, shadow_color, 5, cv2.LINE_AA, tipLength=0.2)

					# Draw the main arrow with gradient layers to simulate depth
					thickness_values = [ x * 3 for x in [4,3,2,1] ] 
					num_layers = len(thickness_values)
					for i in range(num_layers):
						alpha = i / num_layers
						layer_color = tuple(int((1 - alpha) * color[j] + alpha * 255) for j in range(3))  # Blend color towards white
						cv2.arrowedLine(
							image_original, vector_start_point, vector_end_point, layer_color, thickness_values[i],
							cv2.LINE_AA, tipLength=0.2
						)
	

				if write_image or frame_idx % save_freq == 0:
					# print( f"frame: {frame_idx}")
					cv2.imwrite(os.path.join(image_output_folder, input_name + f'_{frame_idx}.jpg'), image_original)
				
			out.write(image_original)
			frame_idx += 1
