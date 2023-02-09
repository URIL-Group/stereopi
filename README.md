# stereopi_URIL
Developing repo of stereopi stereovision code 

## cv_calibrate_stereo.py
A new script to calibrate stereo cameras using openCV. This calibration produces a numpy file with information about the intrinsic/extrinsic properties of the cameras and their rectification. Much more effective calibration than previous script.

## cv_image_disparity.py
This script takes in a left and right image to rectify and produce a disparity map from. Mainly for tuning parameters to use later for stereovideo.

## calibration_parameters_v1.npz
This file of resulting calibration parameters applies to first iteration of stereopi housing set up on land. Data collected with this set up includes Bermuda 05/2022 and Panama 06/2022

## calibration_parameters_tank.npz
This file is the current working calibration parameters for the first iteration of stereopi housing. The calibration parameters were computed after taking several images in the URI Bay Campus Acoustic Tank and running said images through cv_calibrate_stereo.py. 
