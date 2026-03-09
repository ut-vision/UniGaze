import sys
import os
import csv
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from unigaze.gazelib.gaze.normalize import read_image, normalize, estimateHeadPose, normalize_woimg
from unigaze.gazelib.gaze.gaze_utils import vector_to_pitchyaw, angular_error, pitchyaw_to_vector
from unigaze.gazelib.utils.h5_utils import add, to_h5
from unigaze.gazelib.label_transform import lm68_to_50, get_eye_nose_landmarks, get_eye_mouth_landmarks, \
				mean_eye_nose, mean_eye_mouth, get_face_center_by_nose, get_face_center_by_mouth
from unigaze.gazelib.draw.draw_image import draw_lm, draw_gaze

### ============================================================
### the loading functions for XGaze
### ============================================================
def read_image(img_path, camera_matrix, camera_distortion):
	# load input image and undistort
	img_original = cv2.imread(img_path)
	img = cv2.undistort(img_original, camera_matrix, camera_distortion)
	return img

def read_xml(xml_path):
	if not os.path.isfile(xml_path):
		print('no camera calibration file is found.')
		## instead of exit, return an exception
		raise FileNotFoundError("No camera calibration file is found.")
	try:
		fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
		camera_matrix = fs.getNode('Camera_Matrix').mat() # camera calibration information is used for data normalization
		camera_distortion = fs.getNode('Distortion_Coefficients').mat()
		camera_translation = fs.getNode('cam_translation').mat()
		camera_rotation = fs.getNode('cam_rotation').mat()
	except:
		print('the bad xml file is: ', xml_path)
		
	return camera_matrix, camera_distortion, camera_translation, camera_rotation

def read_csv_as_dict(csv_path):
	"""
	Given an annotation file of one subject, store the annotation in a dictionary
		args: the path of the annotation file of xgaze, for example: ETH-XGaze/data/annotation_train/subject0001.csv
		return: a dictionary, whose key is the frame index/camera index, for example: 'frame0000/cam00.JPG' 
				and value is the corresponding annotation
	"""
	with open(csv_path, newline='') as csvfile:
		data = csvfile.readlines()
	reader = csv.reader(data)
	sub_dict = {}
	for row in reader:
		frame = row[0]
		cam_index = row[1]
		sub_dict[frame+'/'+cam_index] = row[2:]
	return sub_dict
	
def read_lm_gc_new(sub_dict, index):
	"""index is e.g. frame0001/cam00.JPG"""
	gaze_point_screen = np.array([int(float(i)) for i in sub_dict[index][0:2]])
	gaze_point_cam = np.array([float(i) for i in sub_dict[index][2:5]])
	head_rotation_cam = np.array([float(i) for i in sub_dict[index][5:8]])
	head_translation_cam = np.array([float(i) for i in sub_dict[index][8:11]])
	lm_2d = np.array([int(float(i)) for i in sub_dict[index][11: 11+136]]).reshape(68,2)
	lm_2d_proj = np.array([int(float(i)) for i in sub_dict[index][11+136:]]).reshape(50,2)

	return  gaze_point_cam, head_rotation_cam, head_translation_cam, lm_2d, lm_2d_proj





### ============================================================
### the loading functions for MPIIFaceGaze
### ============================================================
def read_txt_as_dict(text_path):
	with open(text_path) as f:
		data = f.readlines()
	reader = csv.reader(data)
	p = {}
	for row in reader:
		words = row[0].split()
		p[words[0]] = words[1:]
	return p
def read_lm_gc_for_mpii(person_dict, index):
	landmarks = np.array([int(i) for i in person_dict[index][2:14]]).reshape((6,2))
	gc = np.array([float(i) for i in person_dict[index][23:26]]).reshape((3,1))
	return landmarks, gc
def load_facemodel(model_path):
    # load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
    fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_READ)
    face_model = fs.getNode('face_model').mat()
    return face_model