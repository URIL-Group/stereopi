# StereoPi script to go through calibration photos and save L/R for calibration #
# Alexandra Runyan, July 2022 #

## Imports
import cv2
import os

## Go through folder of photos
total_photos = 0
dir  = "/home/pi/Pictures"
for path in os.listdir(dir):
	if os.path.isfile(os.path.join(dir, path)):
		total_photos += 1
print("...total photos:", total_photos, "...")


## Global variables
photo_height = 480
photo_width = 1280
img_height = 480
img_width = 640
photo_counter = 0

## Main pair cut cycle
print("...Begin photo cycle...")

if (os.path.isdir("./pairs_land_calib_v2")==False):
	os.makedirs("./pairs_land_calib_v2")
while photo_counter != total_photos:
	for filename in os.listdir(dir):
		photo_counter += 1
		f = os.path.join(dir, filename)
		if os.path.isfile(f) == False:
			print("No file named ", f)
			continue
		img_pair = cv2.imread(f, -1)
		cv2.imshow("Image pair", img_pair)
		cv2.waitKey(0)
		img_right = img_pair[0:img_height, 0:img_width]
		img_left = img_pair[0:img_height, img_width:photo_width]
		right_name = "./pairs_land_calib_v2/right_"+str(photo_counter).zfill(2)+".png"
		left_name = "./pairs_land_calib_v2/left_"+str(photo_counter).zfill(2)+".png"
		cv2.imwrite(right_name, img_right)
		cv2.imwrite(left_name, img_left)
		print("...Pair number ", str(photo_counter), "saved...")

print("...End photo cycle...")
