## Cut stereo image pairs into left and right image
## Alexandra Runyan, June 2022

# Imports
import cv2

# Photo pair
pair = "/home/pi/stereopi_URIL/example_pics/box_picture2.jpg"

# Read image, get size, cut into left and right
img_pair = cv2.imread(pair)
#height,width,colors = img_pair.shape
#print(img_pair.shape)
#half_width = width/2
photo_height = 480
photo_width = 1280
img_height = 480
img_width = 640

imgRight = img_pair[0:img_height, 0:img_width]
imgLeft = img_pair[0:img_height, img_width:photo_width]
rightName = "/home/pi/stereopi_URIL/example_pics/box_picture2_right.jpg"
leftName = "/home/pi/stereopi_URIL/example_pics/box_picture2_left.jpg"

# Save right/left images
cv2.imwrite(rightName, imgRight)
cv2.imwrite(leftName, imgLeft)
