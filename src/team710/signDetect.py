print("Import Advance module")
import cv2
import numpy as np
import tensorflow as tf
import glob
# from keras.models import load_model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session

# Ros libraries
import roslib
import rospy
import rospkg

rospack = rospkg.RosPack()
path = rospack.get_path('team710')
    
path = path + '/src/team710/'
# path = path + '/'
print ('PATH = ', path)
print ('in signDetect.py')

# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True

# sess = tf.Session(config=config)

# sess = tf.Session()
# config = tf.ConfigProto(
#     device_count={'GPU': 1},
#     intra_op_parallelism_threads=1,
#     allow_soft_placement=True
# )

# commented here
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
graph = tf.get_default_graph()
set_session(sess)

my_model = load_model(path + 'models/sign_fix.h5')

# my_model = load_model(path + 'models/sign.h5')



def getSignColor():
    '''
    files = glob.glob("sign/*.jpg")
    files = sorted(files)
    print('All file in lane folder: ')
    for file in files:
        print(file)
    '''

    #files = glob.glob("/home/hoc/catkin_ws/src/omni_fun/src/omni_fun/lane/*.jpg")
    files = glob.glob(path + 'sign/*.jpg')
    files = sorted(files)
    print('All file in sign folder: ')
    for file in files:
        print(file)

    sign_crops = [cv2.imread(file) for file in files]

    sign_colors = []
    for crop in sign_crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        average = hsv.mean(axis=0).mean(axis=0)
        sign_colors.append(average)
    return sign_colors


sign_colors = getSignColor()
# set_session(sess)
# sign_model = loadModel(path + 'models/sign_fix.h5')
crop = np.zeros((64, 64), np.uint8)


sign_cascade = cv2.CascadeClassifier(path + 'models/cascade2.xml')


def FindSign(img):
    signs = sign_cascade.detectMultiScale(img,
                                          scaleFactor=1.03,
                                          minNeighbors=2,
                                          minSize=(19, 19),
                                          maxSize=(60, 60),
                                          flags=0)
    return signs


import timeit
from datetime import timedelta
# set_session(sess)
def Cascade_sign(img_):
    global crop, sess, graph
    t = 2
    side = None
    img = img_[0:100, ...].copy()
    img2 = img_[0:100, ...].copy()
    signs = FindSign(img2)
    position = None
    for (x, y, w, h) in signs:
        if x > 5 and y > 5 and x + w < 475 and y + h < 95 and 1.0 * h / w > 0.85 and 1.0 * h / w < 1.3:
            print(w, h)
            position = [x, y, w, h]
            crop = img[y - t:y + h + t, x - t:x + w + t, :]
            crop = cv2.resize(crop, (64, 64))
            start = timeit.default_timer()
            with graph.as_default():
                set_session(sess)
                _side, confident = my_model.predict(np.array([crop]) /
                                                    255.0)[0]
                print('toi dang o day')
                print('confident = ', confident)
            stop = timeit.default_timer()
            print ('Thoi gian chay model la ', stop - start)
            #_side, confident = sign_model.predict(np.array([crop])/255.0)[0]
            # _side = 0
            # confident = 0.9
            if _side < 0.5:
                _side = 0
            else:
                _side = 1
            if confident > 0.5:
                side = _side
            # cv2.rectangle(img,(x-t,y-t),(x+w+t,y+h+t),(0,255,0),2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img,str(confident),(x-10,y-10), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            # #cv2.imshow('crop', crop)
    return crop, side, img, position


def testVideo():
    cap = cv2.VideoCapture(path + "video/video.avi")
    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))

        crop, side, img2, position = Cascade_sign(frame)

        if (side !=  None):
            '''
            x, y, w, h = position
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            text = ''
            if (side == 0):

                text = 'LEFT'
            elif (side == 1):
                text = 'RIGHT'
            cv2.putText(frame,text,(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            '''

        #cv2.imshow('crop', crop)
        #cv2.imshow('sign', img2)
        #cv2.imshow('image', frame)

        # print 'side= ', side
        Key = cv2.waitKey(30)
        if (side != None):
            cv2.waitKey(0)

        if (Key == 27):
            break
        if (Key == ord('s')):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def testImage():
    img = cv2.imread('images/sign1.png')

    crop, side, img2 = FindSign(img)
    #cv2.imshow('crop', crop)
    #cv2.imshow('sign', img2)
    #cv2.imshow('image', img)
    print('side= ', side)
    #getMask(img)

    cv2.waitKey(0)


# testVideo()
