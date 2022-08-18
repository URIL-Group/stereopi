# StereoPi Calibration Script Using OpenCV
# Alexandra Runyan, URIL, July 2022

## Imports
import cv2
import os
import numpy as np

## Directory for Images
pairs_dir = "./pairs_land_calib_v1"

## Define Checkerboard Parameters
CHECKERBOARD = (6,9)
world_scale = 1.094 #size of checkerboard square in inches
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## Define world coordinates
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp = objp * world_scale

## Define vectors for image points
img_pts_L = []
img_pts_R = []
obj_pts = []

## Initialize photo pair counter
total_photos = 0
for filename in os.listdir(pairs_dir):
	total_photos += 1

total_pairs = total_photos/2
print("...Total photo pairs: ", total_pairs, "...")

## Calibration of individual cameras
pair_counter = 0
print("...Start reading image pairs...")

while pair_counter != total_pairs:
	pair_counter += 1
	print("...Import pair number:", pair_counter, "...")
	left_name = pairs_dir+'/left_'+str(pair_counter).zfill(2)+'.png'
	right_name = pairs_dir+'/right_'+str(pair_counter).zfill(2)+'.png'
	if os.path.isfile(left_name) and os.path.isfile(right_name):
		img_left = cv2.imread(left_name)
		img_right = cv2.imread(right_name)
		img_left_gray = cv2.imread(left_name, 0)
		img_right_gray = cv2.imread(right_name, 0)
		outputL = img_left.copy()
		outputR = img_right.copy()
		retL, cornersL = cv2.findChessboardCorners(img_left_gray, CHECKERBOARD, None)
		retR, cornersR = cv2.findChessboardCorners(img_right_gray, CHECKERBOARD, None)

		if retL and retR:
			obj_pts.append(objp)
			#refine checkerboard corners
			cv2.cornerSubPix(img_left_gray, cornersL, (11,11), (-1,-1), criteria)
			cv2.cornerSubPix(img_right_gray, cornersR, (11,11), (-1,-1), criteria)
			cv2.drawChessboardCorners(outputL, CHECKERBOARD, cornersL, retL)
			cv2.drawChessboardCorners(outputR, CHECKERBOARD, cornersR, retR)
			#show checkerboard corner detection
			cv2.imshow("left image", outputL)
			cv2.imshow("right image", outputR)
			cv2.waitKey(0)
			#append points to vectors
			img_pts_L.append(cornersL)
			img_pts_R.append(cornersR)
print("...Finished pair cycle...")

## Calibrate left camera --> get intrinsic cam matrix, distortion coeff, rotation vector (3x1),
## translation vector (3x1)
print("...Begin calibration of individual cameras...")

#imgSize parameter!
imgSize = img_left_gray.shape[1::-1]

retL, int_mtrx_L, distL, rot_vec_L, trns_vec_L = cv2.calibrateCamera(obj_pts, img_pts_L, imgSize, None, None)
htL, wdL = img_left_gray.shape[:2]
new_int_mtrx_L, roiL = cv2.getOptimalNewCameraMatrix(int_mtrx_L, distL, (wdL, htL), 1, (wdL, htL))
print("Per pixel reprojection error left camera:", retL)

## Calibrate right camera
retR, int_mtrx_R, distR, rot_vec_R, trns_vec_R = cv2.calibrateCamera(obj_pts, img_pts_R, imgSize, None, None)
htR, wdR = img_right_gray.shape[:2]
new_int_mtrx_R, roiR = cv2.getOptimalNewCameraMatrix(int_mtrx_R, distR, (wdR, htR), 1, (wdR, htR))
print("Per pixel reprojection error right camera:", retR)

## Calibrate stereo pair --> FIX INTRINSIC MATRICES
print("...Begin calibration of stereo pair...")

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#calculate essential and fundamental matrices between two cameras
retS, new_int_mtrx_L, distL, new_int_mtrx_R, distR, rot, trans, EMat, FMat = cv2.stereoCalibrate(obj_pts, img_pts_L, img_pts_R, new_int_mtrx_L, distL, new_int_mtrx_R, distR, 
imgSize, criteria_stereo, flags)
print("Stereo per pixel reprojection error:", retS)

## Stereo Rectification --> map from L/R images
rectify_scale = 1
rect_L, rect_R, proj_mtrx_L, proj_mtrx_R, Q, roi_L, roi_R = cv2.stereoRectify(new_int_mtrx_L, distL, new_int_mtrx_R, distR, imgSize, rot, trans, rectify_scale, (0,0))

## Compute mapping between rectified images
left_stereo_map = cv2.initUndistortRectifyMap(new_int_mtrx_L, distL, rect_L, proj_mtrx_L, imgSize, cv2.CV_16SC2)

right_stereo_map = cv2.initUndistortRectifyMap(new_int_mtrx_R, distR, rect_R, proj_mtrx_R, imgSize, cv2.CV_16SC2)

## Save parameters using numpy
print("...Saving parameters...")
np.savez_compressed("calibration_parameters_v1.npz", imageSize = imgSize, leftMapX = left_stereo_map[0], leftMapY = left_stereo_map[1], leftROI = roi_L, rightMapX = right_stereo_map[0], 
rightMapY = right_stereo_map[1], rightROI = roi_R)
print("...Parameters saved!...")


