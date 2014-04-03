import numpy as n
from matplotlib import pyplot as plot

a = n.zeros((10,10), n.uint8)

#horizontal line from 1,1 to 8,1
a[1,1:8] = 1

#vertical line from 1,1 to 1,4
a[1:5,1] = 1

#horizontal line from 1,4 to 8,4
a[5,1:8] = 1

#verical line from 8,1 to 8,4
a[1:6,8] = 1

#horizontal line from 1,7 to 8,7
a[7,1:9] = 1

#a[3,5] = 1

print a

## Global Variables for Setting thresholds and limits for comparing the boundaries
origin = (3,5)
## Maximum length of the polar rho axis being detected, this corresponds to the diagonal distance for the image size
maxlength = n.ceil(n.sqrt(n.square(10)+n.square(10)))
print "maxlength:", maxlength
## Maximum angular sweep that is being detected by the accumulator when doing the Hough transform
maxangle = 180
print "maxangle:", maxangle
## The angular threshold for the steps in which the angles need to be iterated upon
angular_threshold = 10
print "angular threshold:", angular_threshold
## The length threshold which is the minimum length of line that we are trying to identify in the given image
length_threshold = 4
print "length threshold:", length_threshold
# Used for increasing sensitivity of the rho value so that values are not integers
# example 1.5 and 1 are not the same and same array item is not incremented since 
# indices come only in integers
rho_resolution = 5
print "rho resolution:", rho_resolution
theta_resolution = 5
print "theta resolution:", theta_resolution
negative_axis = 2
mid_axis = maxlength * rho_resolution 
theta_tolerance = 4
rho_tolerance = 4
length_tolerance = 2
# The accumulator where the values are incremented depending upon the value of Rho obtained for each angle
accum = n.zeros((maxlength*rho_resolution,maxangle*theta_resolution), n.int32)

def getMod(i,j):
    k = i - j
    if k < 0:
        return k * -1
    else:
        return k

def getRho (x, y, theta):
    theta = n.deg2rad(theta)
    rho = ((x) * n.cos(theta) + (y) * n.sin(theta))
    return rho

def getRhoIndex(rho):
    if rho < 0:
        return rho * -1 * rho_resolution
    else:
        return (rho * rho_resolution) + mid_axis

def getThetaIndex(theta):
    return theta * theta_resolution

def getRhoValueFromIndex(index):
    if index > mid_axis:
        return (index - mid_axis) / rho_resolution
    else:
        return (index / rho_resolution) * -1

def getThetaValueFromIndex(index):
    return index / theta_resolution

def displayImage(matrix):
    plot.imshow(matrix, cmap="gray")
    plot.show();
        
def getExtendedPeak(point1,point2):
    return [((point1[0]+point2[0])/2),((point1[1]+point2[1])/2)]

def checkThetaTolerance(point1,point2):
    theta_diff = getMod(point1[1],point2[1])
    if theta_diff < theta_tolerance:
        return True
    else:
        return False

def checkRhoTolerance(point1,point2):
    rho_diff = getMod(point1[0],point2[0])
    if rho_diff < rho_tolerance:
        return True
    else:
        return False

def checkLengthTolerance(point1,point2):
    length1 = accum[point1[0],point1[1]]
    length2 = accum[point2[0],point2[1]]
    length_diff = getMod(length1,length2)
    if length_diff < length_tolerance:
        return True
    else:
        return False

def areLinesPerpendicular(theta1,theta2):
    theta = getMod(theta1,theta2)
    if theta < 90 + theta_tolerance and theta > 90 - theta_tolerance:
        return True
    else:
        return False

def compareLineEquations(line_eq1,line_eq2):
    if line_eq1[0] == line_eq2[0] and line_eq1[1] == line_eq2[1]:
        return True
    else:
        return False
    
def getRelatedPairs(lines):
    counter = 0
    size = lines.shape[0]
    for x in range(0, size):
        theta1 = lines[x][1]
        for y in range(0, size):
            theta2 = lines[y][1]
            for z in range(0, size):
                theta3 = lines[z][1]
                if areLinesPerpendicular(theta1,theta2) and areLinesPerpendicular(theta1,theta3):
                    length = accum[lines[x][0]*rho_resolution,lines[x][1] * theta_resolution]
                    distance = getMod(lines[y][0],lines[z][0])
                    if length < distance + length_tolerance and length > distance - length_tolerance:
                        print "(",lines[x],")",length,"--",distance,"(",lines[y],",",lines[z],")"

def getRelatedPairs2(k_set, l_set):
    result = {}
    size = len(l_set)
    for p in k_set:
        for v in range(0,size):
            for w in range(v,size):
                length = accum[p[0]*rho_resolution,p[1]*theta_resolution]
                distance = getMod(l_set[v][0],l_set[w][0])
                if distance > length - length_tolerance and distance < length + length_tolerance:
                    if p not in result:
                        result[p] = [tuple(l_set[v]),tuple(l_set[w])]
                    else:
                        result[p] = result[p] + [tuple(l_set[v]),tuple(l_set[w])]
    return result
                             
                    
def getParallelPairs(array):
    length = array.shape[0]
    done = []
    parallels = {}
    for x in range(0,length):
        key = array[x][1]
        if key not in done:
            done.append(key)
            parallels[key] = []
            parallels[key].append(tuple(array[x]))
        else:
            parallels[key].append(tuple(array[x]))
    print parallels
    return parallels

def getPerpendicularSets(xarray):
    for t in xarray:
        for m in xarray:
            if getMod(t,m) == 90:
                size_t = len(xarray[t])
                size_m = len(xarray[m])
                point_set = getRelatedPairs2(xarray[t],xarray[m])
                points = compareDimensions(point_set)
                getCoordinates(points)

def getCoordinates(points):
    for g in points:
        for h in points:
            if g[1] != h[1]:
                print "intersection of:",g,"and",h

def removeDuplicates(items):
    res = {}
    for c in items:
        res[tuple(c)] = 1
    res_array = []
    for x,y in res.iteritems():
        res_array.append([x[0],x[1]])
    return n.asarray(res_array,n.int32)

def compareDimensions(point_set):
    rect = {}
    keys = []
    points = []
    for p,q in point_set.iteritems():
        keys.append(p)
        
    size = len(keys)    
    for m in range(0,size):
        for n in range(0, size):
            if m != n and matchAllPoints(point_set[keys[m]],point_set[keys[n]]):
                rect[keys[m]] = 1
                rect[keys[n]] = 1
                for p in point_set[keys[m]]:
                    rect[p] = 1
    for o in rect:
        points.append(o)
    return points

def matchAllPoints(set1,set2):
    v = [x for x in set1 if x not in set2] + [x for x in set2 if x not in set1]
    if len(v) == 0:
        return True
    else:
        return False
    
def checkRectangle(lines):
    P = getParallelPairs(lines)
    getPerpendicularSets(P)
    #P = getRelatedPairs(lines)
    #if compareDimensions(P[0],P[1]) and compareDimensions(P[1],P[0]):
    #print "It is a rectangle!"

def computePeaks(lines):
    r = lines.shape
    peaks = n.zeros((3,2), n.int32)
    for k in range(0,r[0]):
        for j in range(k,r[0]):
            if( k != j and checkThetaTolerance(lines[k],lines[j]) and checkRhoTolerance(lines[k],lines[j]) and checkLengthTolerance(lines[k],lines[j])):
                peak = getExtendedPeak(lines[k],lines[j])
                length1 = accum[lines[k][0] * rho_resolution, lines[k][1] * theta_resolution]
                length2 = accum[lines[j][0] * rho_resolution, lines[j][1] * theta_resolution]
                lines[k] = peak
                lines[j] = peak
                laverage = (length1+length2)/2
                accum[peak[0] * rho_resolution, peak[1] * theta_resolution] = laverage
    lines = removeDuplicates(lines)
    checkRectangle(lines)
                
def main():
    pixels = n.column_stack(n.nonzero(a))
    for p in pixels:
        for theta in range(0,maxangle,10):
            rho = getRho(p[1],p[0],theta)
            abs_rho = getRhoIndex(rho)
            abs_theta = getThetaIndex(theta)
            accum[rho * rho_resolution,abs_theta] = accum[rho * rho_resolution,abs_theta] + 1

    lines = n.column_stack(n.where(accum > length_threshold))
    for v in lines:
        length = accum[v[0],v[1]]
        rho = v[0] = (v[0]/rho_resolution)
        theta = v[1] = getThetaValueFromIndex(v[1])
        print "rho:",rho,"theta:",theta ,"deg", "length:",length
    computePeaks(lines)

main()
    
#plot.imshow(a, cmap="gray")
#plot.show()

