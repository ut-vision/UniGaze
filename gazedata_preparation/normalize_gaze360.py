
import h5py
import argparse
import cv2
import glob
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from datetime import datetime
import face_alignment
from gazelib_interface import *



def str2array(string):
	return np.array(string.split(',')).astype('float')


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

def write_error(text, out_txt_path):
	with open(out_txt_path, 'a') as f:
		f.write( text + '\n')




def convert_gaze3d(gaze3d_read):
	"""
	Converts the 3D vector to definition.
	We empirically found the workflow:
		1. Negate Z 
		2. Negate the whole vector to get the vector of Face center -> Target.
	"""
	gaze3d = gaze3d_read.copy()
	gaze3d *= np.array([1,1,-1]) # first negate the z axis

	vector3d_target_to_face = gaze3d
	vector3d_face_to_target = vector3d_target_to_face * np.array([-1,-1,-1])
	return vector3d_face_to_target





if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--gaze360_intermediate_dir", type=str, help='directory of gaze360 intermediate data')
	parser.add_argument("--focal_norm", type=int, default=860, help='focal length used for normalization')
	parser.add_argument("--distance_norm", type=int, default=320, help='distance used for normalization')
	parser.add_argument("--roi_size", type=int, default=448, help='roi size used for normalization')
	parser.add_argument("--save_size", type=int, default=224, help='the size of the saved normalized image, may require resizing after normalization')
	parser.add_argument('--output_dir', type=str,help='directory of target domain dataset')
	config, _ = parser.parse_known_args()


	base_dir = osp.dirname(osp.abspath(__file__))
	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
	except:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


	face_model_path =  f"{base_dir}/face_model.txt"

	use_50 = True

	focal_norm = config.focal_norm
	distance_norm = config.distance_norm
	roi_size = (config.roi_size, config.roi_size)
	save_size = (config.save_size, config.save_size)


	imroot = config.gaze360_intermediate_dir # '/media/jqin/disk2/Datasets/GAZE_datasets/Gaze360/gaze360_intermediate'
	path = f'{imroot}/Label/'
	save_dir = config.output_dir ##'/media/jqin/disk2/Datasets/GAZE_datasets/Gaze360/gaze360_intermediate/h5_subjects'
	os.makedirs( save_dir, exist_ok=True)

	

	for tag in [ 'train', 'val', 'test']:
		save_folder = os.path.join(save_dir, tag)
		os.makedirs(save_folder, exist_ok=True)

		sample_dir = osp.join(save_folder, 'samples')
		os.makedirs(sample_dir, exist_ok=True)

		error_log = osp.join( osp.dirname(save_folder), 'error_log.txt')
		with open( path + f"{tag}.label") as infile:
			lines = infile.readlines()
			print("Number of lines: ",len(lines))
			header = lines.pop(0)
			print("Header: ", header)
			# Face: train/Face/7371.jpg  0.6257810692361899,-0.38593671970246585,-0.6778280768534976 0.7454940407952673,-0.39622299367145264
			# Left: train/Left/7371.jpg
			# Right: train/Right/7371.jpg
			# Origin: rec_001/head/000000/001620.jpg
			# 3DGaze: 0.6257810692361899,-0.38593671970246585,-0.6778280768534976 
			# 2DGaze: 0.7454940407952673,-0.39622299367145264

			for idx, line in enumerate(tqdm(lines[:])):
				face_path = os.path.join(imroot, 'Image', line.split(' ')[0])
				image = cv2.imread(face_path) # (224,224,3)
				camera_matrix, camera_distortion = set_dummy_camera_model( image = image )

				original_name = line.split(' ')[3]
				rec = original_name.split('/')[0]
				subject_id = original_name.split('/')[2]

				img_name = os.path.join( *(original_name.split('/')[1:]) )  
				gaze3d = line.split(' ')[4]; gaze3d = str2array(gaze3d)
				

				"""
				In their definition, gaze3d (their)  --------GazeTo2d------> gaze2d [yaw, pitch]

				We need to first convert their gaze3d to our convention, and then apply R on the vector,
				Then we can use the vector_to_pitchyaw(-gaze_normalized.reshape((1,3))) to get the 2d [pitch, yaw] gaze in our convention, consistent with other datasets.
				"""


				## -------------------------------------------- landmarks detection --------------------------------------------
				preds = fa.get_landmarks(image)
				if preds is not None:
					## each pred is a detected face
					if len(preds)>1:
						write_error( f'{face_path}/{img_name} has more than one face detected', error_log)
						continue
					
					landmarks = preds[0] # array (68,2)
					landmarks = np.asarray(landmarks)
				else:
					print(f'{face_path}/{img_name} has no face detected')
					write_error( f'{face_path}/{img_name} has no face detected', error_log)
					continue
				lm_gt = landmarks.copy()
				## --------------------------------------------  estimate head pose --------------------------------------------
				face_model_load = np.loadtxt(face_model_path)
				''' Use 50 landmarks to estimate head pose, or only use 6 landmarks '''
				if use_50:
					hr, ht = estimateHeadPose(lm68_to_50(lm_gt).astype(float).reshape(50, 1, 2) , face_model_load.reshape(50, 1, 3), camera_matrix, camera_distortion, iterate=True)
				else:
					face_model = get_eye_nose_landmarks(face_model_load) # the eye and nose landmarks
					landmarks_sub = get_eye_nose_landmarks(lm_gt) # lm_gt[[36, 39, 42, 45, 31, 35], :]

					hr, ht = estimateHeadPose(landmarks_sub.astype(float).reshape(6, 1, 2), face_model.reshape(6, 1, 3), camera_matrix, camera_distortion, iterate=True)
				# # -------------------------------------------------------------------------------------------------------------------
				# compute estimated 3D positions of the landmarks
				ht = ht.reshape((3,1))
				hR = cv2.Rodrigues(hr)[0] # rotation matrix
				face_center_by_nose, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)
				

				######################################################################################################
				### NOTE: Important, convert gaze3d to "FaceCenter-to-Target" vector
				gaze3d = convert_gaze3d(gaze3d)
				######################################################################################################
				gc = face_center_by_nose + gaze3d.reshape((3,1))
				# -------------------------------------------- normalize image --------------------------------------------
				img_face, R, hR_norm, gaze_normalized, landmarks_norm, W = normalize(image, lm_gt, focal_norm, distance_norm, roi_size, face_center_by_nose, hr, ht, camera_matrix, gc=gc)
				hr_norm_pitchyaw = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
				gaze_norm_pitchyaw = vector_to_pitchyaw(-gaze_normalized.reshape((1,3))).flatten()

				img_face = cv2.resize(img_face, save_size)
				landmarks_norm *= save_size[0]/roi_size[0]

				if idx < 50: ## just for comparing the image before and after normalization
					gaze2d = line.split(' ')[5]; gaze2d = str2array(gaze2d) ## read this just for visualization comparison
					image_crop = draw_gaze(image, gaze2d[[1,0]] )
					img_face2 = draw_gaze(img_face, gaze_norm_pitchyaw)
					image_crop = cv2.resize(image_crop, img_face2.shape[:2])
					cv2.imwrite(osp.join(sample_dir, f'img_{idx}.jpg'), cv2.hconcat([ image_crop, img_face2]))
					print('write to  ', osp.join(sample_dir, f'img_{idx}.jpg') )

				to_write = {}
				add(to_write, 'face_patch', img_face.astype(np.uint8))
				add(to_write, 'face_head_pose', hr_norm_pitchyaw)
				add(to_write, 'face_gaze', gaze_norm_pitchyaw)
				add(to_write, 'face_gaze_3d', gaze_normalized)
				add(to_write, 'img_name', str.encode(img_name))
				to_h5(to_write,  os.path.join(save_folder, '{}.h5'.format(subject_id)) )