import cv2
import numpy as np

# Define your color in BGR format (OpenCV default, e.g., pure red)
# For RGB (255, 0, 0), BGR would be (0, 0, 255)
bgr_color = np.uint8([[[48, 70, 200]]]) 

# Convert the BGR color to HSV
hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)

print("HSV value:", hsv_color)

# The output will be an array like [[[  0 255 255]]], 
# indicating the exact H, S, and V values for that color in OpenCV's range.
166, 115, 55
184, 126, 61

200, 70, 48
200, 70, 48