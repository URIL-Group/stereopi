# StereoPi Tune Disparity Parameters Script #
# Alexandra Runyan, URIL, 2022 #
#
#
# ISSUES: for some reason light/dark still means far/close respectively. why?

## Imports ##
import cv2
import numpy as np
import os
from stereovision.calibration import StereoCalibration
from matplotlib import pyplot as plt

## Global Variables
SWS = 45  #search window size & used for block size
PFS = 55 #pre filter size
PFC = 63  #pre filter cap
MDS = 0 #min disparity
NOD = 48 #number of disparities
TTH = 10  #texture threshold
UR = 0   #uniqueness ratio
SR = 0   #speckle range
SPWS = 0 #speckle window size

## Load L/R Images & convert to numpy arrays##
image_left = cv2.imread("./example_pics/bottle_left.jpg", cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread("./example_pics/bottle_right.jpg", cv2.IMREAD_GRAYSCALE)
#image_left = cv2.imread("./rectifyed_left.jpg", cv2.IMREAD_GRAYSCALE)
#image_right = cv2.imread("./rectifyed_right.jpg", cv2.IMREAD_GRAYSCALE)
print("...Showing left image...")
cv2.imshow("left image", image_left)
cv2.waitKey(0)
print("...Showing right image...")
cv2.imshow("right image", image_right)
cv2.waitKey(0)

## Input Calibration Data and Rectified Pair ##
calibration = StereoCalibration(input_folder = "calibration_result_land")
rectified_pair = calibration.rectify((image_left, image_right))
print("...Showing left rectified image...")
cv2.imshow("left image", rectified_pair[0])
cv2.waitKey(0)
print("...Showing right rectified image...")
cv2.imshow("right image", rectified_pair[1])
cv2.waitKey(0)

## Initialize Stereo Block Matcher ##
block_matcher = cv2.StereoBM.create(blockSize = SWS)
#set bm parameters
block_matcher.setPreFilterType(1)
block_matcher.setPreFilterSize(PFS)
block_matcher.setPreFilterCap(PFC)
block_matcher.setMinDisparity(MDS)
block_matcher.setNumDisparities(NOD)
block_matcher.setTextureThreshold(TTH)
block_matcher.setUniquenessRatio(UR)
block_matcher.setSpeckleRange(SR)
block_matcher.setSpeckleWindowSize(SPWS)
disparity = block_matcher.compute(rectified_pair[0], rectified_pair[1])
normalized = disparity/255
#show normalized version of image
print("...Showing normalized disparity...")
cv2.imshow("normalized disparity", normalized)
#cv2.imwrite("disparity1.jpg", normalized) <-- not working to save the disparity image
cv2.waitKey(0)
