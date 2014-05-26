import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot

print "====== This is an experiment for shape detection ====="
img = ocv.imread("dropdown.png",ocv.IMREAD_GRAYSCALE)
template = ocv.imread("tab.png", ocv.IMREAD_GRAYSCALE)
#======= Template Matching ========
methods = ['ocv.TM_CCOEFF','ocv.TM_CCOEFF_NORMED','ocv.TM_CCOR']
matching_method = eval(methods[1])
#====== Reading template elements ======
input_box = ocv.imread("inputbox.png",ocv.IMREAD_GRAYSCALE)
check_box = ocv.imread("checkbox.png", ocv.IMREAD_GRAYSCALE)
menu = ocv.imread("menu.png", ocv.IMREAD_GRAYSCALE)
radio_button = ocv.imread("radio.png", ocv.IMREAD_GRAYSCALE)
button = ocv.imread("button.png", ocv.IMREAD_GRAYSCALE)
tab = ocv.imread("tab.png", ocv.IMREAD_GRAYSCALE)
drop_down = ocv.imread("dropdown.png",ocv.IMREAD_GRAYSCALE)
slider = ocv.imread("slider.png", ocv.IMREAD_GRAYSCALE)
#========================================
element = input_box
width, height = element.shape
match = ocv.matchTemplate(img,element,matching_method)
min_val,max_val, min_loc, max_loc = ocv.minMaxLoc(match)
top_left = max_loc
bottom_right = (top_left[0] + width, top_left[1]+height)
ocv.rectangle(img, top_left, bottom_right, 0 , 2)
#plot.imshow(img,cmap="gray")
#plot.show()
print width,height,min_val, max_val, min_loc, max_loc, top_left, bottom_right

#====== feature detection trial =======
dst = ocv.cornerHarris(img,2,3,0.04)
dst = ocv.dilate(dst,None)

#plot.imshow(dst)
#plot.show()

