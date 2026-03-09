import os
import os.path as osp
import argparse
import math
import numpy as np
import imageio
import scipy.io
import glob
import cv2
import matplotlib.pyplot as plt
import h5py
import time
from tqdm import tqdm
from rich.progress import track
import face_alignment
from gazelib_interface import *



def process_eyediap_person(person_id, config, fa, face_model_50lm):
	data_dir = osp.join(config.eyediap_raw_dir, 'Data', person_id)
	anno_base = osp.join(annotation_dir, person_id)
	
	# Load Gaze State (Find OK frames)
	gaze_state = {}
	with open(osp.join(anno_base, 'gaze_state.txt'), 'r') as f:
		for line in f.readlines():
			words = line.strip().split('\t')
			if len(words) == 2:
				gaze_state[int(words[0])] = (words[1] == 'OK')

	# Load Gaze Target (Ball or Screen)
	gaze_target = {}
	gaze_fname = 'screen_coordinates.txt' if osp.exists(osp.join(data_dir, 'screen_coordinates.txt')) else 'ball_tracking.txt'
	
	# Correction for Ball Tracking to Camera Space
	rot_corr = np.eye(3); rot_corr[1, 1] = rot_corr[2, 2] = -1.0
	tra_corr = np.array([0.0, 0.0, 1.0])

	with open(osp.join(data_dir, gaze_fname), 'r') as f:
		lines = f.readlines()[1:]
		for line in lines:
			words = line.split(';')
			t = np.array([float(words[-3]), float(words[-2]), float(words[-1])], dtype=np.float64)
			if np.allclose(t, 0.0): continue
			
			if gaze_fname == 'ball_tracking.txt':
				t = np.dot(rot_corr.T, t - tra_corr)
			
			gaze_target[int(words[0])] = t * 1e3 # Convert to mm

	# Load Calibration
	# print("opening calibration file: ", osp.join(data_dir, 'rgb_vga_calibration.txt'))
	with open(osp.join(data_dir, 'rgb_vga_calibration.txt'), 'r') as f:
		lines = f.readlines()
		fx, _, cx = [float(d) for d in lines[3].split(';')]
		_, fy, cy = [float(d) for d in lines[4].split(';')]
	camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
	dist_coeffs = np.zeros(5) # EYEDIAP VGA is usually pre-undistorted



	# Process Video
	video_path = osp.join(data_dir, 'rgb_vga.mov')
	cap = cv2.VideoCapture(video_path)
	
	output_h5 = osp.join(config.output_dir, f"{person_id}.h5")

	
	# Sort frames to process linearly
	valid_frames = sorted([f for f in gaze_state.keys() if gaze_state[f] and f in gaze_target and f % config.frame_step == 0])

	for frame_idx in track(valid_frames, description=f"Processing {person_id}"):
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		ret, frame = cap.read()
		if not ret: 
			continue
		img = frame # BGR
		gc = gaze_target[frame_idx].reshape((3, 1))
		# Landmark Detection
		lm68_dets = fa.get_landmarks(img[:,:,::-1])
		if lm68_dets is None:
			continue
		
		lm68 = lm68_dets[0]
		lm50 = lm68_to_50(lm68)
		
		# Head Pose Estimation
		facePts = face_model_50lm.reshape(50, 1, 3).astype(np.float32)
		landmarks_sub = lm50.reshape(50, 1, 2).astype(np.float32)
		hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, dist_coeffs, iterate=True)
		
		# Calculate Face Center for Normalization
		hR = cv2.Rodrigues(hr)[0]
		Fc = np.dot(hR, face_model_50lm.T) + ht
		face_center, _ = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_50lm)
		norm_list = normalize(img, lm50, config.focal_norm, config.distance_norm,
							  (config.roi_size, config.roi_size), face_center, hr, ht, camera_matrix, gc)
		
		img_face, R_mat, hR_norm, gaze_norm, lm_norm = norm_list[0], norm_list[1], norm_list[2], norm_list[3], norm_list[4]
		
		hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
		gaze_pitchyaw = vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten()

		if save_size != roi_size:
			img_face = cv2.resize(img_face, save_size)
			lm_norm = lm_norm / np.array(roi_size) * np.array(save_size)


		to_write = {}
		add(to_write, 'face_patch', img_face.astype(np.uint8) )
		add(to_write, 'face_gaze', gaze_pitchyaw.astype(np.float32))
		add(to_write, 'face_head_pose', hr_norm.astype(np.float32))
		add(to_write, 'face_mat_norm', R_mat.astype(np.float32))
		add(to_write, 'landmarks_norm', lm_norm )
		to_h5(to_write, output_h5)

		# # Visualization
		# if visualize_count < 5:
		# 	vis_dir = f'{output_dir}/vis'
		# 	os.makedirs(vis_dir, exist_ok=True)
		# 	vis_path = osp.join(vis_dir, f"{person_id}_{frame_idx}.jpg")
		# 	img_vis = draw_gaze(img_face.copy(), gaze_pitchyaw)
		# 	cv2.imwrite(vis_path, img_vis)
		# 	visualize_count += 1

	cap.release()




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--eyediap_raw_dir", type=str, help='directory of eyediap raw data')
	parser.add_argument("--frame_step", type=int, default=15, help='frame step for processing')
	parser.add_argument("--focal_norm", type=int, default=960, help='focal length used for normalization')
	parser.add_argument("--distance_norm", type=int, default=600, help='distance used for normalization')
	parser.add_argument("--roi_size", type=int, default=224, help='roi size used for normalization')
	parser.add_argument("--save_size", type=int, default=224, help='the size of the saved normalized image, may require resizing after normalization')
	parser.add_argument('--output_dir', type=str,help='directory of target domain dataset')
	config, _ = parser.parse_known_args()

	base_dir = osp.dirname(osp.abspath(__file__))
	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
	except:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

	### Normalization parameters
	focal_norm = config.focal_norm
	distance_norm = config.distance_norm
	roi_size = (config.roi_size, config.roi_size)
	save_size = (config.save_size, config.save_size)

	face_model_50lm = np.loadtxt( f"{base_dir}/face_model.txt")

	os.makedirs(config.output_dir, exist_ok=True)

	annotation_dir = osp.join(config.eyediap_raw_dir, 'Annotations/GazeState/GazeStateExport/Data')
	person_list = [os.path.basename(p) for p in sorted(glob.glob(annotation_dir + '/*'))]
	print('total persons: ', len(person_list))

	for person_id in person_list[:]:
		process_eyediap_person(person_id, config, fa, face_model_50lm)
