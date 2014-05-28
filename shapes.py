#===== Importing utility libraries for opencv =====
import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot
#========= Importing XML libraries =========
import xml.etree.ElementTree as XML


print "====== This is an experiment for shape detection ====="
img = ocv.imread("dropdown.png",ocv.IMREAD_GRAYSCALE)
template = ocv.imread("tab.png", ocv.IMREAD_GRAYSCALE)

#======= Template Matching Methods ========

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

#================ Function Declarations ========================
    
def addComponent(parent, component, data):
    c = XML.SubElement(parent,component)
    c.set("left", data[0])
    c.set("top", data[1])
    c.set("right", data[2])
    c.set("bottom", data[3])
    c.set("width", data[4])
    c.set("height", data[5])
    return parent
    
def createDocument():
    doc = XML.Element("screen")
    comment = XML.Comment("Elements found in the prototype sketch")
    doc.append(comment)
    return doc
    
def matchImage(image,template):
    height, width = template.shape
    match = ocv.matchTemplate(image,template,matching_method)
    min_val, max_val, min_loc, max_loc = ocv.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1]+height)
    return [top_left, bottom_right, width, height]

#==================== Main function calls =====================
elements  = ["input_box","check_box","radio_button","menu","button","tab","slider"]
doc = createDocument()
for e in elements:
    coords = matchImage(img,eval(e))
#    ocv.rectangle(img, coords[0], coords[1], 0 , 2)
    addComponent(doc, e, [str(coords[0][0]),str(coords[0][1]),str(coords[1][0]),str(coords[1][1]), str(coords[2]), str(coords[3])])

#XML.dump(doc)
#plot.imshow(img,cmap="gray")
#plot.show()


#====== feature detection trial =======
#dst = ocv.cornerHarris(img,2,3,0.04)
#dst = ocv.dilate(dst,None)
grayscale = ocv.cvtColor(img,ocv.COLOR_BGR2GRAY)
corners = ocv.goodFeaturesToTrack(grayscale, 200, 0.005, 0.04)
corners = np.int0(corners)
print corners.shape

for c in corners:
    x,y = c.ravel()
    ocv.circle(img, (x,y),3,255,-1)

plot.imshow(img)
plot.show()




