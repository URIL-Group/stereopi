# StereoPi Image Disparity Calculation Script Using OpenCV
# Alexandra Runyan, URIL, July 2022

## Imports
import cv2
import numpy as np
import os

## Global Variables
#PFS = 27 #pre filter size
PFC = 29  #pre filter cap
BS = 11 #block size; rec:3-11, odd
MDS = -32 #min disparity
NOD = 160 #number of disparities; must be divisable by 16
#TTH = 10  #texture threshold
#UR = 7   #uniqueness ratio; rec:5-15
SR = 16   #speckle range; VALUES MULTIPLIED BY 16
SPWS = 200 #speckle window size; rec:50-200

## Load L/R Images & convert to numpy arrays
print("...loading image pair...")
image_left = cv2.imread("./pairs_tank_calib/left_01.png")
image_right = cv2.imread("./pairs_tank_calib/right_01.png")
#cv2.imshow("left img raw",image_left)

## Load calibration parameters
print("...loading calibration parameters...")
calibration = np.load("calibration_parameters_tank.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

## Rectify images using calibration parameters and show images
fixed_left = cv2.remap(image_left, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
fixed_right = cv2.remap(image_right, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
#print("...showing rectified images...")
#cv2.imshow("left image", fixed_left)
#cv2.imshow("right image", fixed_right)
#cv2.waitKey(0)

## Check image rectification by overlaying images
out = fixed_right.copy()
out[:,:,0] = fixed_right[:,:,0]
out[:,:,1] = fixed_right[:,:,1]
out[:,:,2] = fixed_left[:,:,2]
print("...showing rectified image overlay...")
cv2.imshow("Image overlay", out)
cv2.waitKey(0)

gray_img_L = cv2.cvtColor(fixed_left, cv2.COLOR_BGR2GRAY)
gray_img_R = cv2.cvtColor(fixed_right, cv2.COLOR_BGR2GRAY)
#print("...showing gray rectified images...")
#cv2.imshow("gray L image", gray_img_L)
#cv2.imshow("gray R image", gray_img_L)

## Create block matcher, set tuning variables, visualize disparity
print("...creating block matcher...")
stereoBM = cv2.StereoSGBM_create()
#stereoBM.setPreFilterType(1)
#stereoBM.setPreFilterSize(PFS)
stereoBM.setPreFilterCap(PFC)
stereoBM.setBlockSize(BS)
stereoBM.setMinDisparity(MDS)
stereoBM.setNumDisparities(NOD)
#stereoBM.setTextureThreshold(TTH)
#stereoBM.setUniquenessRatio(UR)
stereoBM.setSpeckleRange(SR)
stereoBM.setSpeckleWindowSize(SPWS)

print("...visualize disparity...")
disparity = stereoBM.compute(gray_img_L, gray_img_R)
disparity = cv2.normalize(disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)
local_max = disparity.max()
local_min = disparity.min()
disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

## Show disparity map
cv2.imshow("Disparity Map", disparity_color)
cv2.waitKey(0)
