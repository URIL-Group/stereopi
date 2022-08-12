# StereoPi Calibration Script
# Alexandra Runyan, Undersea Robotics and Imaging Laboratory (URIL), 2022

## Imports ##
import cv2
import os
import numpy as np
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError

## Directories ##
pairs_dir = "./pairs_land_calib"

## Global Variables Preset ##
# chessboard parameters
rows = 6
columns = 9
square_size = 1.25 #inches

# initialize photo pair counter
total_photos = 0
for filename in os.listdir(pairs_dir):
	total_photos += 1

total_pairs = total_photos/2
print("...Total photo pairs: ", total_pairs, "...")


## Calibration ##
pair_counter = 0
print("...Start reading image pairs...")

while pair_counter != total_pairs:
	pair_counter += 1
	print("...Import pair number:", pair_counter, "...")
	left_name = pairs_dir+'/left_'+str(pair_counter)+'.png'
	right_name = pairs_dir+'/right_'+str(pair_counter)+'.png'
	if os.path.isfile(left_name) and os.path.isfile(right_name):
		img_left = cv2.imread(left_name)
		img_right = cv2.imread(right_name)
		width, height, _ = img_left.shape
		img_size = (width, height)
		calibrator = StereoCalibrator(rows, columns, square_size, img_size) #initialize calibrator
		try:
			calibrator._get_corners(img_left)
			calibrator._get_corners(img_right)
		except ChessboardNotFoundError as error:
			print(error)
			print("...Pair number ", pair_counter, " ignored...")
		else:
			calibrator.add_corners((img_left, img_right), True)

print("...Finished reading image pairs...")

print("...Starting calibration, please wait...")
calibration = calibrator.calibrate_cameras()
calibration.export("calibration_result_land")

## Show rectified pair and average error ##
calibration = StereoCalibration(input_folder = "calibration_result_land")
avg_error = calibrator.check_calibration(calibration)
print("...Average error of calibration object:", avg_error, "...")

rectified_pair = calibration.rectify((img_left, img_right))
cv2.imshow('Left CALIBRATED', rectified_pair[0])
cv2.imshow('Right CALIBRATED', rectified_pair[1])
cv2.imwrite("rectifyed_left.jpg",rectified_pair[0])
cv2.imwrite("rectifyed_right.jpg",rectified_pair[1])
cv2.waitKey(0)

