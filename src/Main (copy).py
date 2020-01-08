#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.
This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time
import math

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
import rospkg 

#tesorflow
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session


# Ros Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

from team710.carControl import getAngle, PID, getBirdView, isCommingToCrossroad
from team710.signDetect import Find_Sign, Cascade_sign
from team710.getMaskByCanny import getLaneMaskByCanny
from team710.getMaskByColor import getMergeMask

sess = tf.Session()
graph = tf.get_default_graph()

rospack = rospkg.RosPack()
path = rospack.get_path('team710')
    
path = path + '/src/'
print 'PATH video = ', path
#outVideo = cv2.VideoWriter(path + 'out2D.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (200,150))

def getDirect(image, _side):
    img = image[80:240,:]
    mask = getLaneMaskByCanny(img)
    # #cv2.imshow('imageprocess', img)
    birdView = getBirdView(mask)
    merge = getMergeMask(img)
    # #cv2.imshow('color mask', merge)
    avg, out, score = getAngle(birdView, _side)
        
    if (score < 80):
        merge = getMergeMask(img)
        birdView2 = getBirdView(merge)
        avg2, out2, score2 = getAngle(birdView2, _side)

        # print 'canny Score= ', score, '    HSV score= ', score2

        # #cv2.imshow('Canny mask', mask)
        # #cv2.imshow('Canny direct', out)
        # #cv2.imshow('Canny birdview', birdView)

        # #cv2.imshow('HSV mask', merge)
        # #cv2.imshow('HSV direct', out2)
        # #cv2.imshow('HSV birdview', birdView2)

        if (score < score2):
            score = score2
            out = out2
            mask = merge
            birdView = birdView2
            avg = avg2
            cv2.putText(image,'Using mask color',
                    (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image,'Using mask Canny',
                    (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image,'Using mask Canny',
                (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

    print 'score= ', score

    return avg, score, mask, out, birdView, image

VERBOSE=False
sideTurn = None
counter = 0
avgSign = 0
turn = False
counter_turn = 0
frame_turn = 30
last_angle = 0
CarSpeed = 50.0
baseSpeed = 50.0
frame_from_crossroad = 0 #so frame ke tu khi phat hien nga tu dung de tang toc lai neu k re~
is_counting_frame = False
side = None
position = None
crossroad = False
c_ima = 0

# cv2.namedWindow('BirdView', cv2.WINDOW_NORMAL)
# cv2.namedWindow('FindAngle', cv2.WINDOW_NORMAL)
# cv2.namedWindow('rgb_image', cv2.WINDOW_NORMAL)

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.speed_publisher = rospy.Publisher("/team710/set_speed", Float32)
        self.angle_publisher = rospy.Publisher("/team710/set_angle", Float32)
        self.camera_angle__publisher = rospy.Publisher("/team710/set_camera_angle", Float32)

        # subscribed Topic
        self.rgb_subscriber = rospy.Subscriber("/team710/camera/rgb/compressed",
            CompressedImage, self.rgb_callback,  queue_size = 1)

        # self.depth_subscriber = rospy.Subscriber("/team1/camera/depth/compressed",
            #CompressedImage, self.depth_callback,  queue_size = 1)

        if VERBOSE :
            print "subscribed to /camera/image/compressed"
        
        

    
    def rgb_callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        
        global sideTurn, counter, avgSign, turn, counter_turn, frame_turn, path
        global is_counting_frame, frame_from_crossroad, last_angle, CarSpeed, baseSpeed
        global side, position, c_ima, crossroad

        

        angle = Float32()
        speed = Float32()
        cam_angle = Float32()

        cam_angle.data = 0

        CarSpeed = baseSpeed

        Key = cv2.waitKey(10)
        if (Key == ord('s')):
            c_ima+=1
            cv2.imwrite( path + str(c_ima) + '.jpg' , image_np)
            cv2.putText(image_np,'Saved', 
                        (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

        
        # img = image_np[80:240,:]
        # birdView, im = getBirdView(img)

        # #cv2.imshow('im', im)
        

        side = None
        position = None
        crop, side, img2, position = Cascade_sign(image_np)
        if (is_counting_frame == True):
            CarSpeed = 30
            # crop, side, img2, position = Cascade_sign(image_np)
            frame_from_crossroad += 1
            cv2.putText(image_np,'Detecting Sign',
                        (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)
            if (frame_from_crossroad >= 20):
                CarSpeed = baseSpeed
                is_counting_frame = False
                frame_from_crossroad = 0

        # crop, side, img2, position = Cascade_sign(image_np)
        if (side != None):
            x, y, w, h = position
            cv2.rectangle(image_np,(x,y),(x+w,y+h),(255,0,0),2)
            text = ''
            if (side == 0):
                text = 'LEFT'
            elif (side == 1):
                text = 'RIGHT'
            cv2.putText(image_np,text,(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            counter += 1
            avgSign += side
            #print 'Counter = ', counter
            CarSpeed = 10

        #Khi thu thap du lieu bien bao den khi khong con nhin thay bien bao nua
        if (counter > 0 and side == None):
            if (avgSign <= int(counter/2)):
                sideTurn = 0
            else:
                sideTurn = 1
            print 'counter = ', counter
            print 'avgSign = ', avgSign
            print 'sideTurn = ', sideTurn
            counter = 0
            avgSign = 0
            turn = True
        
        CarAngle = None
        out = None
        score = None
        mask = None
        birdView = None
        image = None
        if (turn == True):
            '''
            if (sideTurn == 0):
                cam_angle.data = -20
            else:
                cam_angle.data = 20
            CarSpeed = 5.0
            '''
            counter_turn += 1
            if (counter_turn <= frame_turn): #re trong n frame
                #print "Turning", sideTurn
                cv2.putText(image_np,'Turning ' + str(sideTurn), 
                            (150,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)
                CarAngle, score, mask, out, birdView, image = getDirect(image_np, sideTurn)
                # CarAngle, out, score = getAngle(birdView, sideTurn)
            else:
                CarAngle, score, mask, out, birdView, image = getDirect(image_np, None)
                # CarAngle, out, score = getAngle(birdView, None) #neu da re het n frame thi di thang
                counter_turn = 0
                CarSpeed = baseSpeed
                turn = False
        else:
            CarAngle, score, mask, out, birdView, image = getDirect(image_np, None)
            # CarAngle, out, score = getAngle(birdView, None)
            '''
            if (abs(CarAngle - last_angle) > 50):
                CarAngle = last_angle
                cv2.putText(image_np,'FollowLAST ', 
                            (130,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            '''
        
        #Khi den nga tu se giam toc do cua xe
        crossroad = isCommingToCrossroad(birdView)
        if (crossroad == True):
            is_counting_frame = True
            frame_from_crossroad = 0

        if (score < 100 or CarAngle > 40):
            is_counting_frame = True
            frame_from_crossroad = 0
        # if (abs(CarAngle - last_angle) > 30):
        #     CarAngle = 0
        # if (score == 0):
            # CarAngle = last_angle
        CarAngle = PID(CarAngle)
        last_angle = CarAngle
        #birdView = getBirdViewLaneMask(image_np)
        
        #avg, out = getAngle(birdView, side=side)
        
        cv2.putText(image_np,'angle= '+ str(math.ceil(CarAngle*10000)/10000), 
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(image_np,'speed= '+ str(CarSpeed), 
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(image_np,'CamAngle= '+ str(cam_angle.data), 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)               
        angle.data = CarAngle
        speed.data = CarSpeed

        
        cam_angle.data = -3
        self.camera_angle__publisher.publish(cam_angle)
        
        self.angle_publisher.publish(angle)
        self.speed_publisher.publish(speed)

        #print 'Angle = ', avg
        # #cv2.imshow('mask', mask)
        # #cv2.imshow('BirdView', birdView)
        # #cv2.imshow('FindAngle', out)

        #outVideo.write(out)

        # #cv2.imshow('rgb_image', image_np)
        
        

    def depth_callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        
        # #cv2.imshow('depth_image', image_np)
        # cv2.waitKey(2)

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main(sys.argv)