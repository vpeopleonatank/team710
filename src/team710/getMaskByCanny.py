import cv2
import numpy as np

def getClosing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 60, 5)
    # #cv2.imshow('edges', edges)
    kernel = np.ones((9,9),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closing

def getDilate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 70, 5)
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    return dilation

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = 0
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def getLineMask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # lines = cv2.HoughLinesP(edges, 5, np.pi/60, 50, np.array([]), minLineLength=10, maxLineGap=10)
    mask = np.zeros_like(gray)
    '''
    if (lines is not None):
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(mask,(x1,y1),(x2,y2),255,12)
    '''
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            # print 'slope= ', slope
            # print 'intercept= ', intercept
            if (abs(slope) > 0.1):
                L = make_coordinates(image, (slope, intercept))
                x1, y1, x2, y2 = L.reshape(4)
                cv2.line(mask,(x1,y1),(x2,y2),255,10)
        
    return mask

def getLaneMaskByCanny(image):
    closing = getClosing(image)
    dilation = getDilate(image)
    # #cv2.imshow('closing', closing)
    # #cv2.imshow('dilation', dilation)
    mask = cv2.bitwise_not(closing, closing)
    mask[dilation > 100] = 0
    # #cv2.imshow('canny value', mask)
    
    '''
    colorMask = getMergeMask(image)
    mask[colorMask > 100] = 255
    #cv2.imshow('merge', colorMask)
    
    line = getLineMask(image)
    #cv2.imshow('lineMask', line)
    mask[line > 100] = 0
    #cv2.imshow('finis', mask)
    '''

    final = np.zeros(mask.shape)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1] # get largest five contour
    print ('area= ',cv2.contourArea(contours[0]))
    if (cv2.contourArea(contours[0]) > 12000):
        M = cv2.moments(contours[0])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print (cX, cY)
        if (abs(cY - 100) < 30):
            cv2.drawContours(final, contours, 0, 255, -1)
    
    # #cv2.imshow('final', final)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(final, kernel, iterations=1)
    
    return dilation

def testImage():
    image = cv2.imread('images/rbg_1.jpg')
    img = image[80:240,:]
    dilation = getLaneMaskByCanny(img)
    #cv2.imshow('img', image)
    #cv2.imshow('mask', dilation)
    cv2.waitKey(0)


