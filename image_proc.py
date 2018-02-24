import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

images = []

DIR = "/home/thomas/Documents/Python/attachments/"

for file in os.listdir(DIR):
    images.append(file)


image = cv2.imread(DIR + images[0], cv2.IMREAD_UNCHANGED)
blue_channel, green_channel, red_channel = cv2.split(image)
alpha_channel = np.ones(blue_channel.shape, dtype=blue_channel.dtype) * 255 # arbitrary alpha channel

image_BGRA = cv2.merge((blue_channel, green_channel, red_channel, alpha_channel))

IMAGE_WIDTH = image_BGRA.shape[1]
IMAGE_HEIGHT = image_BGRA.shape[0]
IMAGE_RESIZE_FACTOR = 5
image_BGRA = cv2.resize(
        image_BGRA, 
        (
                int(IMAGE_WIDTH / IMAGE_RESIZE_FACTOR), 
                int(IMAGE_HEIGHT / IMAGE_RESIZE_FACTOR)
        )
    )

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

def nothing(x):
    pass

# cv2.createTrackbar('A', 'image', 0, 255, nothing)
cv2.createTrackbar('Sigma', 'image', 0, 100, nothing)

# Kernel for dilation
kernel = np.ones((5,5),np.uint8)

while(True):
    
    sigma = float(cv2.getTrackbarPos('Sigma', 'image')) / 100
    median = np.median(image) 
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    edges = cv2.morphologyEx(
                    cv2.dilate(
                            (
                                cv2.Canny(
                                        image_BGRA, 
                                        lower, 
                                        upper
                                    )
                            ), 
                            kernel, 
                            iterations = 4
                        ),
                    cv2.MORPH_CLOSE, 
                    kernel
                )
            
    cv2.imshow('image', edges)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # alpha = cv2.getTrackbarPos('A','image')

    # image_BGRA[:] = [b,g,r,a]

cv2.destroyAllWindows()

'''
MATRIX_COLUMN_COUNT = 1
if (len(images) % MATRIX_COLUMN_COUNT == 0):
    MATRIX_ROW_COUNT = len(images) / MATRIX_COLUMN_COUNT
else:
    MATRIX_ROW_COUNT = len(images) / MATRIX_COLUMN_COUNT + 1
position = 1

for file in images:
    image = cv2.imread(DIR + file, cv2.IMREAD_UNCHANGED)
    
    sigma = 0.33
    # compute the median of the single channel pixel intensities
    v = np.median(image) 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(image, lower, upper)
    # cv2.imshow(file, edges)
    
    plt.subplot(MATRIX_ROW_COUNT, MATRIX_COLUMN_COUNT, position), plt.imshow(edges, cmap = 'gray')
    
    position += 1

plt.show()
	

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

