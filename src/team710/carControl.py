import cv2
import numpy as np
import math
import time

def Fuzzy(error):
  sign = np.sign(error)
  error =abs(error)
  k1 = 0.35
  x = 50
  k2 = 0.55
  angle = 0
  if error<x:
    angle =  k1*error
  else:
    angle = (error-x)*k2+k1*x
  if angle>60:
    angle = 60
  return -angle*sign

error_arr = np.zeros(5)
t = time.time()

def PID(error, p= 1.0, i =0.01, d = 0.2):
  global error_arr, t
  error_arr[1:] = error_arr[0:-1]
  error_arr[0] = error
  P = error*p
  delta_t = time.time() - t
  t = time.time()
  D = (error-error_arr[1])/delta_t*d
  I = np.sum(error_arr)*delta_t*i
  # print 'error = ', error
  # print 'P = ', P
  # print 'I = ', I
  # print 'D = ', D
  # print '--'
  angle = P + I + D
  if abs(angle)>60:
	  angle = np.sign(angle)*60
  return angle
  #return int(round(angle))

v = 5
#r = 3600/d (d = 0->60)

def Score(img,side, v, d, outImg, color):
  #img : 2D grayScale size (200x150)
  #v : Van toc
  #r : ban kinh cung tron quy dao xe
  #d : goc cua xe
  #bat dau tu chinh giua phia duoi cua anh (vi tri xe hien tai)
  if (abs(d) < 1):
    v = 5
  
  r = 1e10
  if (float(d) != 0):
    r = 3600 / (float)(d)

  #print 'r = ', r
  
  x_org = img.shape[1]/2
  y_org = img.shape[0] - 1

  Max = 0
  x_Max = 100

  t = 0
  x = x_org
  y = y_org

  x_last = x
  y_last = y

  Max = 0

  while(True):
    cv2.circle(outImg, (x,y), 1, color , 2)

    x_last = x
    y_last = y  

    t += 0.5
    x = int (x_org + r*(1 - math.cos((float)(t)*(v/r))))
    y = int(y_org - r*math.sin((float)(t)*(v/r)))

    #Di vuot ra bien cua anh
    if (x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]):
      break
    #tu vung sang di vao vung toi
    if (img[y,x] <= 100 and img[y_last,x_last] > 100):
      break

    score = 0
    if (side == None): #straight
      # score = 150-y - abs(100-x)/2
      score = 150-y - abs(100-x)/3
    elif (side == 0): #left
      score = (150-y)/8 + 2*(100-x)
      if (x > 100): #khong duoc re phai khi co bien bao trai
        score = -1
    elif (side == 1): 
      score = (150-y)/8 + 2*(x-100)
      if (x < 100): #khong duoc re trai khi co bien bao phai
        score = -1

    if (Max <  score and img[y,x] > 100):
        Max = score

  return Max



color1 = (0,0,255)
color2 = (0,255,0)
color3 = (255,0,0)
color4 = (255,255,0)
#img is binary image
def getAngle(img, side):

  out = np.copy(img)

  out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
  out[img > 100] = (255,255,255)
  
  Scores = []

  for d in range(5,80,5):
    Max =  Score(img,side, v, d, out, color1)
    Scores.append((Max, d))
    Max =  Score(img,side, v, -d, out, color1)
    Scores.append((Max, -d))
  Max =  Score(img,side, v, 1e-10, out, color1)
  Scores.append((Max, 1e-10))

  '''
  for d in range(1, 21, 2):
    Max =  Score(img,side, v, d, out, color1)
    Scores.append((Max, d))
    Max =  Score(img,side, v, -d, out, color1)
    Scores.append((Max, -d))
  '''
  Scores.sort(reverse = True)
  top_3 = [Scores[i][1] for i in range(0,3,1)]

  for d in top_3:
      Max =  Score(img,side, v, d, out,color2)
  score = None
  #print ('top3', top_3)
  avg = np.mean(top_3, axis=0)
  score =  Score(img,side, v, avg, out,color3)
  if (side != None and Max == 0):
    if (side == 0):
      avg = -50
    else:
      avg = 50
    score = Score(img,side, v, avg, out,color4)
  if (side == None and Max == 0):
    avg = 0
  
  return avg, out, score


def isCommingToCrossroad(mask2D):
    #mask2D birdview, size (200x150)
    mask2D = cv2.resize(mask2D,(200,150))
    y = 75
    x1 = 0
    x2 = 199
    #do den vung duong
    while(mask2D[y,x1] < 100 and x1 < 199):
        x1 += 1
    #do den vung bien
    x2 = x1
    while(mask2D[y,x2] >= 100 and x2 < 199):
        x2 += 1
    d = x2-x1
    scale = (float(d)/200)
   
    if ( scale < 0.4):
        return False
    return True

def birdViewTransform(image):
        # obtain a consistent order of the points and unpack them
        # individually
    rect = np.array([
                    [65, 15],
                    [255, 15],
                    [600, 80],
                    [-180, 80]], dtype="float32")
    tl, tr, br, bl = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    # image = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    warped = cv2.resize(warped, (200, 150))
    # return the warped image
    return warped

def getBirdView(mask):
    kernel = np.ones((15,15),np.uint8)
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    birdView = birdViewTransform(close)
    return birdView




