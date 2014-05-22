import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot

print "====== This is an experiment for shape detection ====="
img = ocv.imread("dropdown.png",ocv.IMREAD_GRAYSCALE)

#======= Template Matching ========
matching_method = ocv.TM_CCOEFF
template = img[0:120,0:120]
match = ocv.matchTemplate(img,template,matching_method)
plot.imshow(match,cmap="gray")
plot.show()
