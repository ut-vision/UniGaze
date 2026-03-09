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




def process_one_person( person_path ):
	visualize_count = 0
	person = person_path.split('/')[-1]
	txt_path = osp.join( osp.join(data_dir, person), (person+'.txt') )
	person_dict = read_txt_as_dict(txt_path)

	day_list = sorted(glob.glob(person_path + '/' + 'day*'))

	for day_path in track(day_list[:]):
		img_list = sorted(glob.glob(day_path + '/' + '*.jpg'))

		for img_path in img_list[:]:
			day = img_path.split('/')[-2]
			img_name =  img_path.split('/')[-1]

			camera_path = osp.join(data_dir, person, 'Calibration/Camera.mat')
			camera = scipy.io.loadmat(camera_path)
			camera_matrix, camera_distortion = camera['cameraMatrix'], camera['distCoeffs']

			img = read_image(img_path, camera_matrix, camera_distortion)
			lm_gt, gc = read_lm_gc_for_mpii(person_dict, osp.join(day,img_name)) # load "ground truth 2D-landmarks" and "gaze target" from dataset

			## ------------------------------------------- Data Normalization -------------------------------------------
			## estimate head pose 
			facePts = face_model.T.reshape(6, 1, 3)
			landmarks_sub = lm_gt.astype(np.float32)  # input to solvePnP function must be float type
			landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape
			hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion, iterate=True)
			## compute estimated 3D positions of the landmarks
			ht = ht.reshape((3,1))
			hR = cv2.Rodrigues(hr)[0] # rotation matrix
			Fc = np.dot(hR, face_model) + ht # (3,6)
			face_center = np.mean(Fc, axis=1).reshape((3, 1))

			## normalize image
			norm_list = normalize(img, lm_gt, focal_norm, distance_norm, roi_size, face_center, hr, ht, camera_matrix, gc)
			img_face, R, hR_norm, gaze_norm, lm_gt_norm = norm_list[0], norm_list[1], norm_list[2], norm_list[3], norm_list[4]
			hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])

			def headpose_50(lm50):
				facePts = face_model_50lm.reshape(50,1,3)
				landmarks_50 = lm50.astype(np.float32).reshape(50, 1, 2)
				hr, ht = estimateHeadPose(landmarks_50, facePts, camera_matrix, camera_distortion, iterate=True)
				norm_list = normalize_woimg(lm_gt, focal_norm, distance_norm, roi_size, face_center, hr, ht, camera_matrix, gc)

				_, R, hR_norm, _, _  = norm_list[0], norm_list[1], norm_list[2], norm_list[3], norm_list[4]
				hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
				return hr_norm
			
			lm68_dets = fa.get_landmarks(img[:,:,::-1])
			if lm68_dets is not None:
				lm68_det = lm68_dets[0]
				lm50_det = lm68_to_50(lm68_det)
				hr_norm = headpose_50(lm50_det)
			else:
				print('no landmarks detected in image: ', img_path)

			
			if save_size != roi_size:
				img_face = cv2.resize(img_face, save_size)
				lm_gt_norm = lm_gt_norm / np.array(roi_size) * np.array(save_size)

			to_write = {}
			gaze_norm = vector_to_pitchyaw(-gaze_norm.reshape((1,3))).flatten()
			add(to_write, 'face_patch', img_face.astype(np.uint8) )
			add(to_write, 'face_gaze', gaze_norm.astype(np.float32))
			add(to_write, 'face_head_pose', hr_norm.astype(np.float32))
			add(to_write, 'face_mat_norm', R.astype(np.float32))
			add(to_write, 'landmarks_norm', lm_gt_norm )
			to_h5(to_write,  osp.join(output_dir, person + '.h5') )

			### visualize
			### only visualize the first image
			if visualize_count < 10:
				vis_dir = f'{output_dir}/vis'
				os.makedirs(vis_dir, exist_ok=True)
				img_vis = img_face.copy()
				img_vis = draw_lm(img_vis, lm_gt_norm, radius=3)
				img_vis = draw_gaze(img_vis, gaze_norm)
				# img_vis = draw_gaze(img_vis, hr_norm, color=(0, 255, 0))
				cv2.putText(img_vis, '{}'.format(hr_norm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				cv2.imwrite( osp.join(vis_dir, 'img_{}_{}.jpg'.format(person, visualize_count)), img_vis)
			visualize_count += 1






if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--mpiifacegaze_raw_dir", type=str, help='directory of mpiifacegaze raw data')
	parser.add_argument("--focal_norm", type=int, default=960, help='focal length used for normalization')
	parser.add_argument("--distance_norm", type=int, default=300, help='distance used for normalization')
	parser.add_argument("--roi_size", type=int, default=448, help='roi size used for normalization')
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


	face_model = load_facemodel(f"{base_dir}/face_model_for_mpii.yml")# (3,6)

	face_model_50lm = np.loadtxt( f"{base_dir}/face_model.txt")

	## set dataset path
	data_dir = config.mpiifacegaze_raw_dir
	output_dir = config.output_dir
	os.makedirs(output_dir, exist_ok=True)

	person_list = sorted(glob.glob(data_dir + '/' + 'p*'))
	for person_path in person_list[:]:
		process_one_person(person_path)
