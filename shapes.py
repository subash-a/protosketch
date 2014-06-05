#===== Importing utility libraries for opencv =====
import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot
#========= Importing XML libraries =========
import xml.etree.ElementTree as XML
#======== Importing Custom Libraries ========
import page_builder as pbuild

#============ Constants =====================
GRAY = ocv.IMREAD_GRAYSCALE
ASIS = ocv.IMREAD_UNCHANGED
COLOR = ocv.IMREAD_COLOR
HESSIAN_THRESHOLD = 400
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 1
TABLE_NUMBER = 6
KEY_SIZE = 12
MULTI_PROBE_LEVEL = 1
TREES = 5
CHECKS = 100

print "====== This is an experiment for shape detection ====="
img = ocv.imread("dropdown.png", ocv.IMREAD_GRAYSCALE)
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
def matchImage(image,template):
    height, width = template.shape
    match = ocv.matchTemplate(image,template,matching_method)
    min_val, max_val, min_loc, max_loc = ocv.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1]+height)
    return [top_left, bottom_right, width, height]


#================= Template matching technique ================
def templateMatching():
    elements  = ["input"
                 , "check_box"
                 , "radio_button"
                 , "menu"
                 , "button"
                 , "tab"    
                 , "slider"]
    jsfiles = ["utils/js/jquery-1.7.1.js","utils/js/bootstrap.js"]
    cssfiles = ["utils/css/bootstrap.css","utils/css/prototype.css"]
    document = pbuild.createDocument()
    for e in elements:
        coords = matchImage(img,eval(e))
        ocv.rectangle(img, coords[0], coords[1], 0 , 2)
        pbuild.addComponent(document,e,[coords[0][1],coords[0][0],coords[1][0],coords[1][1],coords[2],coords[3]])
        
    html = pbuild.createHTMLPage(document,jsfiles,cssfiles)
    pbuild.showXML(html)
    plot.imshow(img,cmap="gray")
    plot.show()


#============= Feature Detection Technique =====================
def readImage(image, flag):
    m = ocv.imread(image, flag)
    return m

def colorChange(image, destColor):
    c = ocv.cvtColor(image, destColor)
    return c

def detectFeatures(image, method):
    if(method == "SIFT"):
        algorithm = ocv.SIFT()
    elif(method == "SURF"):
        algorithm = ocv.SURF(HESSIAN_THRESHOLD)
    elif(method == "FAST"):
        algorithm = ocv.FastFeatureDetector()

    keyPoints = algorithm.detect(image, None)
    return (keyPoints,algorithm)

def showKeyPoints(image,keypoints,flag):
    kp_image = ocv.drawKeypoints(image,keypoints,color=(0,255,255))
    plot.imshow(kp_image)
    plot.show()

def computeDescriptors(image, method):
    keypoints, algorithm = detectFeatures(image,method)
    keypoints, descriptors = algorithm.compute(image,keypoints)
    return keypoints, descriptors
    
def matchFeatures(desc1,desc2,matching_algorithm,isknnmatch,feature_algorithm):
    if(matching_algorithm == "BRUTE_FORCE"):
        algorithm = ocv.BFMatcher(ocv.NORM_L2)
        if(isknnmatch == True):
            matches = algorithm.knnMatch(desc1,desc2,k=2)
        else:
            matches = algorithm.match(desc1,desc2)
    elif(matching_algorithm == "FLANN"):
        if(feature_algorithm == "SIFT" or feature_algorithm == "SURF"):
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = TREES)
        else:
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = TABLE_NUMBER,
                                key_size = KEY_SIZE,
                                multi_probe_level = MULTI_PROBE_LEVEL)
        search_params = dict(checks = CHECKS)
        algorithm = ocv.FlannBasedMatcher(index_params, search_params)
        if (isknnmatch == True):
            matches = algorithm.knnMatch(desc1,desc2,k=2)
        else:
            matches = algorithm.match(desc1,desc2)
    return matches

def featureDetection():
    src_key, src_desc = computeDescriptors(img,"SIFT")
    dest_key, dest_desc = computeDescriptors(input_box,"SIFT")
    match = matchFeatures(src_desc,dest_desc,"BRUTE_FORCE",False,"SIFT")
    for m in match:
        print m
    
#    FLANN_INDEX_KDTREE = 0
#    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#    search_params = dict(checks = 50)
#    
#    flann = ocv.FlannBasedMatcher(index_params, search_params)
#    matches = flann.knnMatch(src_desc, dest_desc, k = 2)
#
#
#    src_pts = np.float32([src_key[m.queryIdx].pt 
#                          for m,n in matches]).reshape(-1,1,2)
#    dest_pts = np.float32([dest_key[m.trainIdx].pt 
#                           for m,n in matches]).reshape(-1,1,2)
#
#    print src_pts
#    print dest_pts
#
#    M,mask = ocv.findHomography(src_pts, dest_pts, ocv.RANSAC, 5.0)
#    print M
#    good = []
#    for m,n in matches:
#        good.append(m)
#
#    matchesMask = mask.ravel().tolist()
#    draw_params = dict(matchColor = (0,255,0), 
#                       singlePointColor = None, 
#                       matchesMask = matchesMask, 
#                       flags = 2)
#    img3 = ocv.drawMatches(img,src_key,input_box,dest_key,good,None,**draw_params)
#    plot.imshow(img3,'gray')
#    plot.show()
#==================== Main function calls =====================
featureDetection()




