

import cv2
import numpy as np
import os
import copy
import random
from matplotlib import pyplot as plt


class Images:

    def __init__(self, image_path):

        if image_path is None:
            current_file = os.path.dirname(__file__)
            self.dir = os.path.join(current_file, "attachments/")
        else:
            self.dir = image_path

        self.images = []
        for f in os.listdir(self.dir):
            self.images.append(f)

        self.image_index = 0

    def get_random_image(self):
        return random.choice(self.images)

    def get_next_image(self):
        next_image = self.images[self.image_index % len(self.images)]
        self.image_index += 1
        return next_image

    def get_raw_image(self):
        return cv2.imread(self.dir + self.get_random_image(), cv2.IMREAD_UNCHANGED)


class ImageDisplay:

    SIGMA = "sigma"
    DELTA = "delta"
    ALPHA = "ON : 0 \n OFF : 1"

    def __init__(self, image_holder, window_name):

        self.image_holder = image_holder

        if not window_name is None:
            self.window_name = window_name
        else:
            self.window_name = "PTS"

    def trackbar_handler(x):
        print(x)

    def build_display(self, image_on_display):

        def click_event(event, x, y, flags, params):

            image = params

            if event == cv2.EVENT_LBUTTONDOWN:
                blue = image[y, x, 0]
                green = image[y, x, 1]
                red = image[y, x, 2]

                self.image_holder.assign_median_color([blue, green, red])

                strRGB = str(red) + "," + str(green) + "," + str(blue)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, strRGB, (x, y), font, 1, (255, 255, 255), 2)
                cv2.imshow(self.window_name, image)

            if event == cv2.EVENT_RBUTTONDOWN:
                self.reset_display()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar(self.SIGMA, self.window_name, 0, 100, self.trackbar_handler)
        cv2.createTrackbar(self.DELTA, self.window_name, 10, 100, self.trackbar_handler)
        cv2.setTrackbarMin(self.DELTA, self.window_name, 10)
        cv2.createTrackbar(self.ALPHA, self.window_name, 0, 1, self.trackbar_handler)

        cv2.setMouseCallback(self.window_name, click_event, image_on_display)
        cv2.imshow(self.window_name, image_on_display)

    def reset_display(self):
        self.build_display(self.image_holder.orginal_image)




class ImageHolder:

    def __init__(self, image):
        self.original_image = copy.deepcopy(image)
        self.current_image = copy.deepcopy(image)

        self.median_color = None

    def assign_median_color(self, median_color):
        self.median_color = median_color

    def get_original_image(self):
        return self.original_image

    def get_current_image(self):
        return self.current_image


''' Code '''

images = Images()

RAW_IMAGE = images.get_raw_image()

BLUE_CHANNEL, GREEN_CHANNEL, RED_CHANNEL = cv2.split(RAW_IMAGE)
ALPHA_CHANNEL = np.ones(BLUE_CHANNEL.shape, dtype = BLUE_CHANNEL.dtype) * 255 # arbitrary alpha channel

image_BGRA = cv2.merge((BLUE_CHANNEL, GREEN_CHANNEL, RED_CHANNEL, ALPHA_CHANNEL))

IMAGE_WIDTH = image_BGRA.shape[1]
IMAGE_HEIGHT = image_BGRA.shape[0]
IMAGE_RESIZE_FACTOR = 5

# Resize the imported picture
image_BGRA = cv2.resize(
    image_BGRA, 
    (
            int(IMAGE_WIDTH / IMAGE_RESIZE_FACTOR), 
            int(IMAGE_HEIGHT / IMAGE_RESIZE_FACTOR)
    )
)




# Kernel for morphological transformations
kernel = np.ones((5,5), np.uint8)

'''
sigma = float(cv2.getTrackbarPos(SIGMA, WINDOW_NAME)) / 100
median = np.median(image_BGRA) 
lower = int(max(0, (1.0 - sigma) * median))
upper = int(min(255, (1.0 + sigma) * median))


edges = cv2.Canny(
            image_BGRA, 
            lower, 
            upper
        )
'''

closed_image = cv2.morphologyEx(image_BGRA, cv2.MORPH_CLOSE, kernel) 
dilated_image = cv2.dilate(image_BGRA, kernel, iterations  = 1)
opened_image = cv2.morphologyEx(image_BGRA, cv2.MORPH_OPEN, kernel)

g_image_on_display = image_BGRA
g_original_image_on_display = copy.deepcopy(g_image_on_display)

while(True):
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    alpha_state = cv2.getTrackbarPos(ALPHA, WINDOW_NAME)
    delta_value = cv2.getTrackbarPos(DELTA, WINDOW_NAME)
    
    if alpha_state == 1 and not (g_median_color is None):
       
        for y in g_image_on_display:
            accumulated_median_color_values = np.sum(g_median_color)
            for x in y:
                blue = g_image_on_display[y, x, 0]
                green = g_image_on_display[y, x, 1]
                red = g_image_on_display[y, x, 2]
                if abs(accumulated_median_color_values - (blue + green + red)) <= (1 / delta_value) * accumulated_median_color_values:
                    g_image_on_display[y, x, 3] = 0
        
        cv2.imshow(WINDOW_NAME, g_image_on_display)

cv2.destroyAllWindows()

