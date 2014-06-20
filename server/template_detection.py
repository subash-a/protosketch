#===== Importing utility libraries for opencv ==================================
import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot
#========= Importing XML libraries =============================================
import xml.etree.ElementTree as XML
#======== Importing Custom Libraries ===========================================
import page_builder as pbuild

#============ OPENCV CONSTANTS =================================================
GRAY = ocv.IMREAD_GRAYSCALE # read image as gray scale image #
ASIS = ocv.IMREAD_UNCHANGED # read image as is ,also include alpha channels #
COLOR = ocv.IMREAD_COLOR # read image as colored image no alpha #
MATCHING_THRESHOLD = 0.95

#======== SIFT PARAMETERS ======================================================
SIFT_NUMBER_OF_FEATURES = 100
SIFT_NUMBER_OF_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESHOLD = 0.04
SIFT_EDGE_THRESHOLD = 10
SIFT_SIGMA = 1.6

#============ SURF PARAMETERS ==================================================
SURF_HESSIAN_THRESHOLD = 400 # The Hessian matrix threshold for SURF #
SURF_NUMBER_OF_OCTAVES = 4
SURF_NUMBER_OF_OCTAVE_LAYERS = 4
SURF_USE_128 = False
SURF_ORIENTATION = True

#============ ORB PARAMETERS ==================================================
ORB_NUMBER_OF_FEATURES = 500
ORB_SCALE_FACTOR = 1.5
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_FIRST_LEVEL = 0
ORB_WTA_K = 2
ORB_SCORE_TYPE = ocv.ORB_HARRIS_SCORE
ORB_PATCH_SIZE = 31

#========== BRUTE FORCE MATCHER PARAMETERS =====================================
FLANN_INDEX_KDTREE = 0 # FLANN based matching algorithm selection(SIFT,SURF) #
FLANN_INDEX_LSH = 1 # FLANN based matching algorithm (ORB,BRIEF) #
TABLE_NUMBER = 6 # Index param value for FLANN_INDEX_LSH #
KEY_SIZE = 12 # Index param value for FLANN_INDEX_LSH #
MULTI_PROBE_LEVEL = 2 # Index param value for FLANN_INDEX_LSH #
TREES = 5 # Index param value for FLANN_INDEX_KDTREE #
CHECKS = 100 # search param value for number of checks #

print "====== This is an experiment for shape detection ====="

#======= Template Matching Methods =============================================

methods = ['ocv.TM_CCOEFF','ocv.TM_CCOEFF_NORMED','ocv.TM_CCOR','ocv.TM_SQDIFF','ocv.TM_SQDIFF_NORMED']
matching_method = eval(methods[1])


#================ Function Declarations ========================================
# Reads Image and returns a matrix of the image as per the flags #
def readImage(image, flag):
    m = ocv.imread(image, flag)
    return m

# Changes the color scheme of the image to one of BGR, GRAY, HSL #
def colorChange(image, destColor):
    c = ocv.cvtColor(image, destColor)
    return c

def matchImage(image,template):
    height, width = template.shape
    match = ocv.matchTemplate(image,template,matching_method)
    min_val, max_val, min_loc, max_loc = ocv.minMaxLoc(match)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1]+height)
    return [top_left, bottom_right, width, height]

def matchMultipleImage(image, template):
    height, width = template.shape
    match = ocv.matchTemplate(image, template, matching_method)
    loc = np.where(match > MATCHING_THRESHOLD)
    locations = []
    for pt in zip(*loc[::-1]):
        tup = (pt, width, height)
        locations.append(tup)
    return locations
    
#====== Reading template elements ==============================================
img = ocv.imread("assets/dropdown.png", ocv.IMREAD_GRAYSCALE)
template = ocv.imread("assets/tab.png", ocv.IMREAD_GRAYSCALE)
input_box = readImage("assets/inputbox.png",GRAY)
check_box = readImage("assets/checkbox.png", GRAY)
menu = readImage("assets/menu.png", GRAY)
radio_button = readImage("assets/radio.png", GRAY)
button = readImage("assets/button.png", GRAY)
tab = readImage("assets/tab.png", GRAY)
drop_down = readImage("assets/dropdown.png",GRAY)
slider = readImage("assets/slider.png", GRAY)
form = readImage("assets/form.png", GRAY)
sample = readImage("assets/SketchedTemplate.png",GRAY)
#================= Template matching technique =================================
def extractComponentFromImage(image, component, document):
    result = matchMultipleImage(image, eval(component))
    for c, width, height in result:
        ocv.rectangle(sample, c,(c[0] + width , c[1] + height),(0,255,0),2)
        pbuild.addComponent(document, component, c, width, height)
    return document
    
def templateMatching():
    elements  = ["input_box"
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
        document = extractComponentFromImage(sample, e, document)
    html = pbuild.createHTMLPage(document,jsfiles,cssfiles)
    ET = XML.ElementTree(html)
    ET.write("output/index.html")


#============= Feature Detection Functions =====================================
# Preprocess Image for better detection of descriptors
def preProcess(image):
    stage_1 = ocv.gaussian(image)
# Detects the key points of a given image using a given algorithm #
def detectFeatures(image, method):
    if(method == "SIFT"):
        algorithm = ocv.SIFT(SIFT_NUMBER_OF_FEATURES
                             , SIFT_NUMBER_OF_OCTAVE_LAYERS 
                             , SIFT_CONTRAST_THRESHOLD
                             , SIFT_EDGE_THRESHOLD
                             , SIFT_SIGMA)
    elif(method == "SURF"):
        algorithm = ocv.SURF(SURF_HESSIAN_THRESHOLD
                             , SURF_NUMBER_OF_OCTAVES
                             , SURF_NUMBER_OF_OCTAVE_LAYERS
                             , SURF_USE_128
                             , SURF_ORIENTATION)
    elif(method == "ORB"):
        algorithm = ocv.ORB(ORB_NUMBER_OF_FEATURES
                            , ORB_SCALE_FACTOR 
                            , ORB_N_LEVELS 
                            , ORB_EDGE_THRESHOLD 
                            , ORB_FIRST_LEVEL 
                            , ORB_WTA_K
                            , ORB_SCORE_TYPE 
                            , ORB_PATCH_SIZE)

    keyPoints = algorithm.detect(image, None)
    return (keyPoints,algorithm)

# Displays the key points of a image given the keypoints #
def showKeyPoints(image,keypoints,flag):
    kp_image = ocv.drawKeypoints(image,keypoints,color=(0,255,255))
    plot.imshow(kp_image)
    plot.show()

# Computes the desciptors from keypoints using the algorithm provided #
def computeDescriptors(image, method):
    keypoints, algorithm = detectFeatures(image,method)
    keypoints, descriptors = algorithm.compute(image,keypoints)
    return keypoints, descriptors

# Matches the descriptor sets using a matching technique give by user #    
def matchFeatures(desc1,desc2,matching_algorithm,isknnmatch,feature_algorithm):
    if(matching_algorithm == "BRUTE_FORCE"):
        if(feature_algorithm == "SIFT" or feature_algorithm == "SURF"):
            distance_type = ocv.NORM_L2
        else:
            distance_type = ocv.NORM_HAMMING
        algorithm = ocv.BFMatcher(distance_type)
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

# Return the matching keypoints from the training image that is found in 
# query image 
# trainIdx is the corresponding matching point from the training index 
# descriptors
def getMatchingKeypoints(match,src_key,dest_key):
    matching_keypoints = []
    for m in match:
        matching_keypoints.append(dest_key[m.trainIdx])
    return matching_keypoints

def showMatchingPoints(src_image,src_matches,dest_image,dest_matches):
    src_kp_image = ocv.drawKeypoints(src_image, src_matches, color=(0,255,255))
    dest_kp_image = ocv.drawKeypoints(dest_image, dest_matches, color=(0,255,255))
#    plot.subplot(2,1,1)
#    plot.imshow(src_kp_image)
#    plot.subplot(2,1,2)
#    plot.imshow(dest_kp_image)
#    plot.show()



def featureDetection():
    FD_METHOD = "SIFT"
    FM_METHOD = "FLANN"
    print "==== Feature Detection ==="
    print "Feature detection method: ",FD_METHOD
    print "===SIFT PARAMETERS==="
    print "SIFT_NUMBER_OF_FEATURES: ",SIFT_NUMBER_OF_FEATURES
    print "SIFT_NUMBER_OF_OCTAVE_LAYERS: ",SIFT_NUMBER_OF_OCTAVE_LAYERS
    print "SIFT_CONTRAST_THRESHOLD: ",SIFT_CONTRAST_THRESHOLD
    print "SIFT_EDGE_THRESHOLD: ",SIFT_EDGE_THRESHOLD
    print "SIFT_SIGMA: ",SIFT_SIGMA
    print "=== Featue Matching ==="
    print "Feature matching methos: ",FM_METHOD
    print "=== FLANN PARAMETERS ==="
    print "FLANN_INDEX_KDTREE: ",FLANN_INDEX_KDTREE
    print "FLANN_INDEX_LSH : ",FLANN_INDEX_LSH 
    print "TABLE_NUMBER : ",TABLE_NUMBER 
    print "KEY_SIZE : ",KEY_SIZE 
    print "MULTI_PROBE_LEVEL : ",MULTI_PROBE_LEVEL 
    print "TREES : ",TREES 
    print "CHECKS : ",CHECKS 
    src_key, src_desc = computeDescriptors(input_box,FD_METHOD)
    dest_key, dest_desc = computeDescriptors(sample,FD_METHOD)
    print "Source Keypoints: ",len(src_key)
    print "Destination Keypoints: ",len(dest_key)
    print "Source Descriptors: ",len(src_desc)
    print "Destination Descriptors: ",len(dest_desc)
    match = matchFeatures(src_desc,dest_desc,FM_METHOD,False,FD_METHOD)
    print "Number of Matches: ",len(match)
    src_indices = []
    dest_indices = []
    for g in match:
        src_indices.append(src_key[g.queryIdx])
        dest_indices.append(dest_key[g.trainIdx])
        print "Training Desc Index:", g.trainIdx, ", Query Desc Index: ", g.queryIdx

    kps = getMatchingKeypoints(match,src_key,dest_key)
    showMatchingPoints(input_box,src_indices,sample,dest_indices)
 #   plot.subplot(2,1,1), plot.imshow(input_box)
 #   plot.subplot(2,1,2), plot.imshow(sample)
 #   plot.show()
 #   src_pts = np.float32([src_key[m.queryIdx].pt 
 #                         for m,n in match]).reshape(-1,1,2)
 #   dest_pts = np.float32([dest_key[m.trainIdx].pt 
 #                          for m,n in match]).reshape(-1,1,2)
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
#==================== Main function calls ======================================
#templateMatching()
featureDetection()



