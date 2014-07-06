#===== Importing utility libraries for opencv ==================================
import cv2 as ocv
import scipy as sp
import numpy as np
from matplotlib import pyplot as plot
import math as math
#========= Importing XML libraries =============================================
import xml.etree.ElementTree as XML
#======== Importing Custom Libraries ===========================================
import page_builder as pbuild

#============ OPENCV CONSTANTS =================================================
GRAY = ocv.IMREAD_GRAYSCALE # read image as gray scale image #
ASIS = ocv.IMREAD_UNCHANGED # read image as is ,also include alpha channels #
COLOR = ocv.IMREAD_COLOR # read image as colored image no alpha #
BIN_THRESH = ocv.THRESH_BINARY
INVBIN_THRESH = ocv.THRESH_BINARY_INV
MEAN_THRESHOLD = ocv.ADAPTIVE_THRESH_MEAN_C
GAUSS_THRESHOLD = ocv.ADAPTIVE_THRESH_GAUSSIAN_C
SCALE_CUBIC = ocv.INTER_CUBIC
SCALE_LINEAR = ocv.INTER_LINEAR
SCALE_AREA = ocv.INTER_AREA
MATCHING_THRESHOLD = 0.95
ERODE_KERNEL_SIZE = (3,3)
DILATE_KERNEL_SIZE = (2,2)
FEATURES_THRESHOLD = 10
#======= THRESHOLDING PARAMETERS ===============================================
IMAGE_THRESHOLD = 187 # Image thresholding parameter for Adaptive Thresholding
IMAGE_THRESHOLD_MAXVAL = 255 #In Image thresholding value to be given for > thr
IMAGE_THRESHOLD_BLOCK_SIZE = 29 #Neighborhood size in which to do thresholding
IMAGE_THRESHOLD_CONST = 5
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

methods = ['ocv.TM_CCOEFF'
           , 'ocv.TM_CCOEFF_NORMED'
           , 'ocv.TM_CCOR'
           , 'ocv.TM_SQDIFF'
           , 'ocv.TM_SQDIFF_NORMED']
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

def loadTemplates():
    checkbox = readImage("assets/components/checkbox.png", GRAY)
    dropdown = readImage("assets/components/dropdown.png", GRAY)
    radio = readImage("assets/components/radio.png", GRAY)
    button = readImage("assets/components/button.png", GRAY)
    tab = readImage("assets/components/tab.png", GRAY)
    slider = readImage("assets/components/slider.png", GRAY)
    sample = readImage("assets/test_images/sketch_10.jpg",GRAY)
    return checkbox,dropdown,radio,button,tab,slider,sample
#================= Template matching technique =================================
def extractComponentFromImage(image, component, document):
    result = matchMultipleImage(image, eval(component))
    for c, width, height in result:
        ocv.rectangle(sample, c,(c[0] + width , c[1] + height),(0,255,0),2)
        pbuild.addComponent(document, component, c, width, height)
    return document
    
def templateMatching():
    elements  = ["checkbox"
                 , "radio"
                 , "dropdown"
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
def erodeImage(image):
    erode_kernel = np.ones(ERODE_KERNEL_SIZE,np.uint8)
    eroded_image = ocv.erode(image,erode_kernel,iterations=1)
    return eroded_image

def dilateImage(image):
    dilate_kernel = np.ones(DILATE_KERNEL_SIZE,np.uint8)
    dilated_image = ocv.dilate(image
                                 , dilate_kernel
                                 , iterations=1)
    return dilated_image

def thresholdImage(image,binary_flag):
    if binary_flag:
        retval, threshold_image = ocv.threshold(image
                                                , IMAGE_THRESHOLD
                                                , IMAGE_THRESHOLD_MAXVAL
                                                , BIN_THRESH)
    else:
        threshold_image = ocv.adaptiveThreshold(image
                                                , IMAGE_THRESHOLD_MAXVAL
                                                , GAUSS_THRESHOLD
                                                , BIN_THRESH
                                                , IMAGE_THRESHOLD_BLOCK_SIZE
                                                , IMAGE_THRESHOLD_CONST)
    return threshold_image

def removeImageNoise(image):
    kernel = np.ones((2,2),np.uint8)
#    i_opened = ocv.morphologyEx(image,ocv.MORPH_OPEN,kernel)
    i_closed = ocv.morphologyEx(image,ocv.MORPH_CLOSE,kernel)
    return i_closed

def smoothImage(image):
    smooth_image = ocv.GaussianBlur(image,(3,3),0,0)
    return smooth_image
# ====== Image transformation ==================================================
def scaleImage(image):
    height, width = image.shape[:2]
    scaled_image = ocv.resize(image,None,fx=0.30,fy=0.30, interpolation = SCALE_AREA)
    return scaled_image

def preProcess(image):
    output = scaleImage(image)
#    output = thresholdImage(output,False)
#    output = dilateImage(output);
#    output = erodeImage(output)
#    output = removeImageNoise(output)
    output = smoothImage(output)
    return output

#Display all the parameters of the given feature extraction method
def showParameters(params):
    for x in params:
        print x,": ",eval(x)

# Detects the key points of a given image using a given algorithm #
def detectFeatures(image, method):
    if(method == "SIFT"):
        algorithm = ocv.SIFT(SIFT_NUMBER_OF_FEATURES
                             , SIFT_NUMBER_OF_OCTAVE_LAYERS 
                             , SIFT_CONTRAST_THRESHOLD
                             , SIFT_EDGE_THRESHOLD
                             , SIFT_SIGMA)
        print "=== SIFT PARAMETERS ==="
        showParameters(["SIFT_NUMBER_OF_FEATURES"
                             , "SIFT_NUMBER_OF_OCTAVE_LAYERS "
                             , "SIFT_CONTRAST_THRESHOLD"
                             , "SIFT_EDGE_THRESHOLD"
                             , "SIFT_SIGMA"])

    elif(method == "SURF"):
        algorithm = ocv.SURF(SURF_HESSIAN_THRESHOLD
                             , SURF_NUMBER_OF_OCTAVES
                             , SURF_NUMBER_OF_OCTAVE_LAYERS
                             , SURF_USE_128
                             , SURF_ORIENTATION)
        print "=== SURF PARAMETERS ==="
        showParameters(["SURF_HESSIAN_THRESHOLD"
                             , "SURF_NUMBER_OF_OCTAVES"
                             , "SURF_NUMBER_OF_OCTAVE_LAYERS"
                             , "SURF_USE_128"
                             , "SURF_ORIENTATION"])

    elif(method == "ORB"):
        algorithm = ocv.ORB(ORB_NUMBER_OF_FEATURES
                            , ORB_SCALE_FACTOR 
                            , ORB_N_LEVELS 
                            , ORB_EDGE_THRESHOLD 
                            , ORB_FIRST_LEVEL 
                            , ORB_WTA_K
                            , ORB_SCORE_TYPE 
                            , ORB_PATCH_SIZE)
        print "=== ORB PARAMETERS ==="
        showParameters(["ORB_NUMBER_OF_FEATURES"
                            , "ORB_SCALE_FACTOR "
                            , "ORB_N_LEVELS "
                            , "ORB_EDGE_THRESHOLD "
                            , "ORB_FIRST_LEVEL "
                            , "ORB_WTA_K"
                            , "ORB_SCORE_TYPE "
                            , "ORB_PATCH_SIZE"])

    keyPoints = algorithm.detect(image, None)
    return (keyPoints,algorithm)

# Displays the key points of a image given the keypoints #
def writeKeypoints(keypoints,filename):
    f = open(filename,'w')
    f.write("x,y,size,angle,response,octave,class_id\n")
    for k in keypoints:
        f.write(str(k.pt[0])+","+str(k.pt[1])+","+str(k.size)+","+str(k.angle)+","+str(k.response)+","+str(k.octave)+","+str(k.class_id)+"\n")
    f.close()

def showKeyPoints(image,keypoints):
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
        print "=== FLANN PARAMETERS ==="
        showParameters(["FLANN_INDEX_KDTREE"
                        , "FLANN_INDEX_LSH"
                        , "TABLE_NUMBER"
                        , "MULTI_PROBE_LEVEL"
                        , "CHECKS"
                        , "TREES"
                        , "KEY_SIZE"])
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
def ratioTest(matches):
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    return good_matches
    
def getMatchingKeypoints(match,src_key,dest_key):
    matching_keypoints = []
    for m in match:
        matching_keypoints.append(dest_key[m.trainIdx])
    return matching_keypoints

def getKnnMatchingKeypoints(match,src_key,dest_key):
    matching_keypoints = []
    for m in match:
        matching_keypoints.append(dest_key[m[0].trainIdx])
    return matching_keypoints

def showMatchingPoints(src_image,src_matches,dest_image,dest_matches):
    src_kp_image = ocv.drawKeypoints(src_image, src_matches, color=(0,255,255))
    dest_kp_image = ocv.drawKeypoints(dest_image, dest_matches, color=(0,255,255))
    plot.subplot(2,1,1)
    plot.imshow(src_kp_image)
    plot.subplot(2,1,2)
    plot.imshow(dest_kp_image)
    plot.show()

def buildComponentCoordinates(component, keypoints):
    points = []
    for p in keypoints:
        points.append(p.pt)
    print points
    points_array = np.vstack(points)
    np.mean(points_array,0)

def getKeypointIndexes(matches,src_keypoints,dest_keypoints):
    src_indices = []
    dest_indices = []
    for g in matches:        
        src_indices.append(src_key[g.queryIdx])
        dest_indices.append(dest_key[g.trainIdx])
        print "Training Desc Index:", g.trainIdx, ", Query Desc Index: ", g.queryIdx
        print "Distance: ",g.distance
    return src_indices, dest_indices
    
def getKnnKeypointIndexes(matches,src_keypoints,dest_keypoints):
    src_indices = []
    dest_indices = []
    for g in matches:        
        src_indices.append(src_keypoints[g[0].queryIdx])
        dest_indices.append(dest_keypoints[g[0].trainIdx])
        print "Training Desc Index:", g[0].trainIdx, ", Query Desc Index: ", g[0].queryIdx
        print "Distance: ",g[0].distance
    return src_indices, dest_indices

def matchMultipleObjects(dest_indices,dest_key):
    m_responses = []
    u_responses = []
    all_matches = []
    for j in dest_indices:
        m_responses.append(j.response)
    for k in dest_key:
        u_responses.append(k.response)
    unique_m_responses = np.unique(np.sort(m_responses))
    for m in unique_m_responses:
        index  = np.where(u_responses == m)
        for i in index[0]:
            all_matches.append(dest_key[i])
    return all_matches

def getKeypointCoordinates(keypoints):
    x_array = []
    y_array = []
    for k in keypoints:
        x_array.append(k.pt[0])
        y_array.append(k.pt[1])
    return x_array, y_array

def getXYVariances(keypoints):
    x_array,y_array = getKeypointCoordinates(keypoints)
    x_diff = []
    y_diff = []
    x_array = np.sort(x_array)
    y_array = np.sort(y_array)
    print x_array
    print y_array
    for i in xrange(1,len(x_array)):
        x_diff.append(x_array[i] - x_array[i-1])
        y_diff.append(y_array[i] - y_array[i-1])
    print "x-max: ",np.nanmax(x_diff)
    print "y-max: ",np.nanmax(y_diff)
    print "x-min: ",np.nanmin(x_diff)
    print "y-min: ",np.nanmin(y_diff)

def getMaxMin(numarray):
    return np.nanmax(numarray),np.nanmin(numarray)

def getWidthHeight(x1,x2,y1,y2):
    return x2-x1,y2-y1

def getWHRatio(width,height):
    return width/height

def enumerateObjects(src_keys,matching_keys):
    # gives the number of features under the source and matched set
    s_numfeatures = len(src_keys)
    m_numfeatures = len(matching_keys)
    # gives the arrays of x coordinates and y coordinates for source and match
    s_xarray, s_yarray = getKeypointCoordinates(src_keys)
    m_xarray, m_yarray = getKeypointCoordinates(matching_keys)
    # gives the max and min X and Y values from the x and y arrays
    s_maxX,s_minX = getMaxMin(s_xarray)
    s_maxY,s_minY = getMaxMin(s_yarray)
    m_maxX,m_minX = getMaxMin(m_xarray)
    m_maxY,m_minY = getMaxMin(m_yarray)
    # gives the width and height of the source and matched feature set
    s_width,s_height = getWidthHeight(s_minX,s_maxX,s_minY,s_maxY)
    m_width,m_height = getWidthHeight(m_minX,m_maxX,m_minY,m_maxY)
    # gives the ratio of the width/height useful in getting proportion and scale
    s_ratio = getWHRatio(s_width,s_height)
    m_ratio = getWHRatio(m_width,m_height)
    
    # gives the ratio of heights and widths to decide scale 
    height_ratio = m_height/s_height
    width_ratio = m_width/s_width
    # gives the arrangement of the objects in the image
    if m_ratio > s_ratio:
        object_arrangement = "Horizontal"
    else:
        object_arrangement = "Vertical"
    # gives the approximate object width and height
    object_width = s_ratio*m_height
    object_height = m_height

    # gives the approximate number of objects
    approx_objects = m_ratio/s_ratio
    object_num = int(approx_objects)
    for x in xrange(0,object_num):
        print "Top and Left of Object: ",m_minY,m_minX+(x*object_width)
def buildXYResponseArray(keys):
    result = []
    for k in keys:
        result.append(np.array((k.pt[0],k.pt[1],k.response)))
    return np.array(result)

def getResponseArray(keys):
    result = []
    for k in keys:
        result.append(k.response)
    return np.array(result)
    
def reducePoints(mul_match_keys,src_keys):
    # extracts the x and y coordinate arrays from keypoints
    x_array,y_array = getKeypointCoordinates(mul_match_keys)
    # extracts the response values as an array from keypoints
    res_array = getResponseArray(mul_match_keys)
    # builds an array of form [x,y,response]
    dataset = buildXYResponseArray(mul_match_keys)
    # returns the indices of a sorted x-coords array
    sortedx = np.argsort(x_array)
    count_array = []
    matches_array = []
    # checks where the response value is same and then groups them into a bin
    # also tracks their count to measure how many objects have been detected
    for s in sortedx:
        matches = np.where(res_array == res_array[s])
        matches_array.append(matches[0])
        count_array.append(len(matches[0]))
    print np.bincount(count_array)
    print np.argmax(np.bincount(count_array))
    print len(mul_match_keys)/len(src_keys)
    # Number of objects as per the matches of features
    objects = np.argmax(np.bincount(count_array))
    i = 0
    # get to the match which has least x value but has same number of matches 
    # as the object
    while len(matches_array[i]) != objects:
        i = i+1
    # store the indices of the matches with same response value
    point_indices = matches_array[i]
    final_points = []
    final_points_y = []
    #return the set of points that form the center of the objects and the ycoord
    for c in point_indices:
        final_points.append(dataset[c])
        final_points_y.append(y_array[c])
    # find the minimun of y axis     
    minY = np.min(y_array)
    #subtract the elements of points from the minimun y axis
    measure = np.subtract(final_points_y,[minY])
    # get the least difference from the remaining list, this tells us which 
    # point is closest to the top, sp we assign the to left to that and then 
    # subtract the corresponding values from other point y axis as well
    least = np.amin(measure)
    
    final_measure = np.subtract(final_points,[0,least,0])
    return final_measure
    
def featureDetection(source,query):
    FD_METHOD = "SURF"
    FM_METHOD = "BRUTE_FORCE"
    src_image = source
    component_name = "button"
    dest_image = preProcess(query)
    print "=== Feature Extraction ==="
    print "Feature extraction method: ", FD_METHOD
    print "=== Featue Matching ==="
    print "Feature matching method: ", FM_METHOD
    src_key, src_desc = computeDescriptors(src_image, FD_METHOD)
    dest_key, dest_desc = computeDescriptors(dest_image, FD_METHOD)
    print "===== Feature Descriptors ===="
    print "Source Keypoints: ", len(src_key)
    print "Destination Keypoints: ", len(dest_key)
    print "Source Descriptors: ", len(src_desc)
    print "Destination Descriptors: ", len(dest_desc)
#    showKeyPoints(src_image,src_key)
#    writeKeypoints(src_key,"output/source_keypoints.csv")
#    writeKeypoints(dest_key,"output/destination_keypoints.csv")
    match = matchFeatures(src_desc,dest_desc,FM_METHOD,True,FD_METHOD)
    print "Number of Matches: ",len(match)
    print "===== Matching Features ====="
    if(len(match) > FEATURES_THRESHOLD):
        src_indices, dest_indices = getKnnKeypointIndexes(match,src_key,dest_key)
        kps = getKnnMatchingKeypoints(match,src_key,dest_key)
        mul_dest_indices = matchMultipleObjects(dest_indices,dest_key)
        writeKeypoints(src_indices,"output/matching_src_keypoints.csv")
        writeKeypoints(mul_dest_indices,"output/multi_matching_dest_keypoints.csv")
        showMatchingPoints(src_image,src_indices,dest_image,mul_dest_indices)
    #    enumerateObjects(src_key,mul_dest_indices)
        attribs = reducePoints(mul_dest_indices,src_key)
        print attribs
        return attribs
    else:
        return []
#templateMatching()

def extractComponentFromImage2(source,query, component, document):
    coords_array = featureDetection(source,query)
    for c in coords_array:
        pbuild.addComponent(document, component, c, None, None)
    return document

def buildPrototype(image):
    checkbox,dropdown,radio,button,tab,slider,sample = loadTemplates()
    elements = ["tab","dropdown","button","radio","checkbox","slider"]
    test_elements  = ["tab"]
    jsfiles = ["utils/js/jquery-1.7.1.js","utils/js/bootstrap.js"]
    cssfiles = ["utils/css/bootstrap.css","utils/css/prototype.css"]
    document = pbuild.createDocument()
    for e in elements:
        document = extractComponentFromImage2(eval(e),image, e, document)
        html = pbuild.createHTMLPage(document,jsfiles,cssfiles)
        
    ET = XML.ElementTree(html)
    ET.write("output/index.html")

def __main__(imagefile):
    buildPrototype(readImage("upload/"+imagefile,GRAY))



