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
import yaml
import time
from tqdm import tqdm
from rich.progress import track
from omegaconf import OmegaConf

from gazelib_interface import *


def process_one_frame(frame_path, sub_dict, calibration_path, save_path):
	
	image_list = sorted(glob.glob(frame_path + '/' + '*.JPG'))
	for input_path in image_list:
		img_name = os.path.basename(input_path) 
		frame = input_path.split('/')[-2]
		subject = input_path.split('/')[-3]

		camera_path = os.path.join(calibration_path , img_name.replace('.JPG','.xml'))
		camera_matrix, camera_distortion, camera_translation, camera_rotation = read_xml(camera_path)


		img_path = os.path.join(raw_path, subject, frame, img_name)
		img = read_image(img_path, camera_matrix, camera_distortion)
		if img_name in ['cam03.JPG', 'cam06.JPG','cam13.JPG']:
			img = cv2.rotate(img, cv2.ROTATE_180)
			
		gc, hr, ht, lm_gt, lm_50_proj = read_lm_gc_new(sub_dict, os.path.join(frame,img_name))

		## ------------------------------------------- Data Normalization -------------------------------------------
		## estimate head pose 
		# compute estimated 3D positions of the landmarks
		ht = ht.reshape((3,1))
		hR = cv2.Rodrigues(hr)[0] # rotation matrix
		face_center_by_nose, Fc_nose = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)
		## normalize image
		img_face, R, hR_norm, gaze_norm, landmarks_norm, W = normalize(img, lm_gt, focal_norm, distance_norm, roi_size, face_center_by_nose, hr, ht, camera_matrix, gc)
		'''
			img_face: is the normalized face
			R: normalization matrix
			hR_norm: normalized head rotation matrix
			gaze_norm: normalized gaze direction (vector form)
			landmarks_norm: the landmarks in the normalized face image
		'''
		hr_norm = np.array([np.arcsin(hR_norm[1, 2]),np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
		
		if save_size != roi_size:
			img_face = cv2.resize(img_face, save_size)
			landmarks_norm = landmarks_norm / np.array(roi_size) * np.array(save_size)
		
		to_write = {}
		gaze_norm = vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten()
		add(to_write, 'face_patch', img_face)
		add(to_write, 'frame_index', int(frame[-4:]) )
		add(to_write, 'cam_index', int(img_name[-6:-4])+1 )
		add(to_write, 'face_gaze', gaze_norm.astype(np.float32))
		add(to_write, 'face_head_pose', hr_norm.astype(np.float32))
		add(to_write, 'face_mat_norm', R.astype(np.float32))
		add(to_write, 'landmarks_norm', landmarks_norm)

		to_h5(to_write, save_path)
		
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--xgaze_raw_dir", type=str, help='directory of xgaze raw data')
	parser.add_argument("--focal_norm", type=int, default=960, help='focal length used for normalization')
	parser.add_argument("--distance_norm", type=int, default=300, help='distance used for normalization')
	parser.add_argument("--roi_size", type=int, default=448, help='roi size used for normalization')
	parser.add_argument("--save_size", type=int, default=224, help='the size of the saved normalized image, may require resizing after normalization')
	parser.add_argument('--output_dir', type=str,help='directory of target domain dataset')
	config, _ = parser.parse_known_args()

	os.makedirs( config.output_dir, exist_ok=True)

	### Normalization parameters
	focal_norm = config.focal_norm
	distance_norm = config.distance_norm
	roi_size = (config.roi_size, config.roi_size)
	save_size = (config.save_size, config.save_size)

	base_dir = osp.dirname(osp.abspath(__file__))
	face_model_load = np.loadtxt( f"{base_dir}/face_model.txt")

	raw_path = os.path.join(config.xgaze_raw_dir, 'data/train') 
	calibration_path = os.path.join(config.xgaze_raw_dir, 'avg_cams_final') 
	annatation_path = os.path.join(config.xgaze_raw_dir, 'data/annotation_updated') 


	subject_list = sorted(glob.glob(raw_path + '/' + 'subject*'))
	print('total subjects: ', len(subject_list))

	for subject_path in subject_list:
		
		frame_list = sorted(glob.glob(subject_path + '/' + 'frame*'))
		subject = subject_path.split('/')[-1]
		csv_path = os.path.join(annatation_path, subject+'_update.csv')
		calibration_subject = os.path.join(calibration_path, subject)
		os.makedirs( os.path.join(config.output_dir,'train'), exist_ok=True)

		config.save_path = os.path.join(config.output_dir,'train', subject+'.h5')
		print( ' config.save_path: ', config.save_path)


		sub_dict = read_csv_as_dict(csv_path)
		for frame_path in track(frame_list[:]):
			print(frame_path + ' of total {} frame '.format(len(frame_list)))
			frame = frame_path.split('/')[-1]
			process_one_frame(frame_path, sub_dict, calibration_subject, save_path=config.save_path)
