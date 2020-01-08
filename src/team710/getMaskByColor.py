import cv2
import numpy as np
import glob
# Ros libraries
import roslib
import rospy
import rospkg 


'''
video source: video/video.avi
images source: images/*.jpg
shadow, snow cut from image (for detect color of shadow, snow, v.v..): cut/*.jpg

'''

# m is an chanel splitted from HSV image
def getAvgAndVariance(m):
    '''
        average = (m1 + m2 + ... + mN) / N
        variance = sum(m(i) - average)^2 / N
    '''
    avg = m.mean()
    temp = np.square(np.subtract(m, avg))
    variance = np.sum(temp) / (temp.shape[0]*temp.shape[1])
    #variance = np.sqrt(variance)
    return avg, variance

#get h_average, h_variance, s_average, s_variance of each lane color
#all lane image file get from folder /lane/*.jpg
files = None
def getColor():
    global files
    rospack = rospkg.RosPack()
    path = rospack.get_path('team710')
    
    path = path + '/src/team710/lane_pub_test/*.jpg'
    print 'PATH = ', path

    files = glob.glob(path)
    files = sorted(files)
    print('All file in lane folder: ')
    for file in files:
        print(file)

    images = [cv2.imread(file) for file in files]
    colors = []
    for img in images:
        blur = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        avg_v = v.mean()
        h_avg, h_variance = getAvgAndVariance(h)
        s_avg, s_variance = getAvgAndVariance(s)
        v_avg, v_variance = getAvgAndVariance(v)
        colors.append((h_avg, h_variance, s_avg, s_variance, avg_v))
    return colors


# m is an chanel splitted from HSV image
#return matrix likelihood of all pixel in chanels
def getChanelLikelihood(m, avg, variance):
    '''
    likelihood = exp(-sqr(m-m_avg) / sqr(m_variance))
    '''
    temp = np.square(m - avg)
    temp = temp / np.square(variance)
    temp = np.exp(-temp)
    return temp

#get likelihood matrix of each chanel
def getLikelihood(hsv_image, lane_color):
    h_avg, h_variance, s_avg, s_variance, avg_v = lane_color

    h_chanel , s_chanel, v_chanel = cv2.split(hsv_image)

    h_likelihood = getChanelLikelihood(h_chanel, h_avg, h_variance)
    s_likelihood = getChanelLikelihood(s_chanel, s_avg, s_variance)
    
    likelihood = (h_likelihood, s_likelihood, avg_v)
    return likelihood

#get mask of lane by likelihood
def getMask(image, colors):
    blur_img = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

    masks = []
    
    h_chanel , s_chanel, v_chanel = cv2.split(hsv_img)

    for color in colors:
        h_likelihood, s_likelihood, avg_v = getLikelihood(hsv_img, color)
        temp = np.zeros((image.shape[0], image.shape[1]))
        temp[h_likelihood*s_likelihood > 0.3] = 1
        temp2 = np.zeros((image.shape[0], image.shape[1]))
        temp2[abs(v_chanel - avg_v) < 50] = 1
        temp = cv2.bitwise_and(temp, temp2)
        masks.append(temp)

    return masks

def preProcess(mask, minsize):
    mask = np.uint8(mask)
    black = np.zeros(mask.shape,np.uint8)
    
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > minsize:
            approx = cv2.approxPolyDP(cnt,2,True)
            cv2.drawContours(black, [approx], -1, 255, -1)
    return black



colors = getColor()
def getMergeMask(image):
    global colors, files
    # image = cv2.resize(image, (320,240))
    # img = image[80:240,:]
    masks = getMask(image, colors)
    merge = np.zeros(masks[0].shape, np.uint8)
    for i in range(len(masks)):
        #print('mask', i)
        masks[i] = preProcess(masks[i], minsize=3500)
        # #cv2.imshow('mask' + files[i], masks[i])
        merge = cv2.bitwise_or(merge, masks[i])

    merge = preProcess(merge, minsize=5000)
    # print 'addedMaskShape', addedMask.shape
    # #cv2.imshow('finish mask', merge)

    return merge


def testImage():
    image = cv2.imread('images/rbg_1.jpg')
    merge = getMergeMask(image)


    #cv2.imshow('view', merge)
    #cv2.imshow('image', image)
    cv2.waitKey(0)


# testImage()