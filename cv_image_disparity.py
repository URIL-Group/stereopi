# StereoPi Image Disparity Calculation Script Using OpenCV
# Alexandra Runyan, URIL, July 2022

## Imports
import cv2
import numpy as np
import os

## Global Variables
PFS = 5 #pre filter size
PFC = 63  #pre filter cap
BS = 11 #block size; rec:3-11, odd
MDS = -3 #min disparity
NOD = 32 #number of disparities; must be divisable by 16
TTH = 10  #texture threshold
UR = 7   #uniqueness ratio; rec:5-15
SR = 3   #speckle range; VALUES MULTIPLIED BY 16
SPWS = 150 #speckle window size; rec:50-200

## Load L/R Images & convert to numpy arrays
print("...loading image pair...")
image_left = cv2.imread("./example_pics/box_picture1_left.jpg")
image_right = cv2.imread("./example_pics/box_picture1_right.jpg")

## Load calibration parameters
print("...loading calibration parameters...")
calibration = np.load("calibration_parameters.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

## Rectify images using calibration parameters and show images
fixed_left = cv2.remap(image_left, leftMapX, leftMapY, cv2.INTER_LINEAR)
fixed_right = cv2.remap(image_right, rightMapX, rightMapY, cv2.INTER_LINEAR)
print("...showing rectified images...")
cv2.imshow("left image", fixed_left)
cv2.imshow("right image", fixed_right)
#cv2.waitKey(0)
gray_img_L = cv2.cvtColor(fixed_left, cv2.COLOR_BGR2GRAY)
gray_img_R = cv2.cvtColor(fixed_right, cv2.COLOR_BGR2GRAY)

## Create block matcher, set tuning variables,  visualize disparity
print("...creating block matcher...")
stereoBM = cv2.StereoSGBM_create()
#stereoBM.setPreFilterType(1)
#stereoBM.setPreFilterSize(PFS)
stereoBM.setPreFilterCap(PFC)
stereoBM.setBlockSize(BS)
stereoBM.setMinDisparity(MDS)
stereoBM.setNumDisparities(NOD)
#stereoBM.setTextureThreshold(TTH)
stereoBM.setUniquenessRatio(UR)
stereoBM.setSpeckleRange(SR)
stereoBM.setSpeckleWindowSize(SPWS)

print("...visualize disparity...")
disparity = stereoBM.compute(gray_img_L, gray_img_R)
local_max = disparity.max()
local_min = disparity.min()
disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

## Show disparity map
cv2.imshow("Disparity Map", disparity_color)
cv2.waitKey(0)
