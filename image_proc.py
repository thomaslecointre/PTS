''' Global Variables '''

g_median_color = []
g_image_on_display = []
g_original_image_on_display = []


''' Imports '''

import cv2
import numpy as np
import os
import copy
from matplotlib import pyplot as plt

''' Functions ''' 

def trackbar_handler(x):
    print(x)
    
def assign_median_color(color_values):
    global g_median_color
    g_median_color = color_values
    
def build_display(image):
    

def reset_image_on_display():
    global g_image_on_display
    g_image_on_display = g_original_image_on_display
    cv2.imshow("image", g_image_on_display)
    


def click_event(event, x, y, flags, params): 
    
    image = params
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        blue = image[y,x,0]
        green = image[y,x,1]
        red = image[y,x,2]
        
        assign_median_color([blue, green, red])
        
        strRGB = str(red) + "," + str(green) + "," + str(blue)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, strRGB, (x,y), font, 1, (255,255,255), 2)
        cv2.imshow("image", image)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        reset_image_on_display() 
        
''' Code '''

images = []
    
CURRENT_FILE = os.path.dirname(__file__)
DIR = os.path.join(CURRENT_FILE, "attachments/")

for file in os.listdir(DIR):
    images.append(file)


RAW_IMAGE = cv2.imread(DIR + images[1], cv2.IMREAD_UNCHANGED)
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

# Name the window
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# Trackbar for canny edge detection (sigma value)
cv2.createTrackbar("Sigma", "image", 0, 100, trackbar_handler)
cv2.createTrackbar("delta", "image", 0, 100, trackbar_handler)
ALPHA = "ON : 0 \n OFF : 1"
cv2.createTrackbar(ALPHA, "image", 0, 1, trackbar_handler)


# Kernel for morphological transformations
kernel = np.ones((5,5), np.uint8)

'''
sigma = float(cv2.getTrackbarPos('Sigma', 'image')) / 100
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

cv2.imshow("image", g_image_on_display)

cv2.setMouseCallback("image", click_event, g_image_on_display)

while(True):
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    alpha_state = cv2.getTrackbarPos(ALPHA, "image")
    delta_value = cv2.getTrackbarPos("delta", "image")
    
    if alpha_state == 1 and not (g_median_color is None):
       
        for y in g_image_on_display:
            accumulated_median_color_values = np.sum(g_median_color)
            for x in y:
                blue = g_image_on_display[y, x, 0]
                green = g_image_on_display[y, x, 1]
                red = g_image_on_display[y, x, 2]
                if abs(accumulated_median_color_values - (blue + green + red)) <= (1 / delta_value) * accumulated_median_color_values:
                    g_image_on_display[y, x, 3] = 0
        
        cv2.imshow("image", g_image_on_display)

cv2.destroyAllWindows()

