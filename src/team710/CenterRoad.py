import math
import numpy as np
import cv2
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
      score = 150-y - abs(100-x)/2
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

  for d in range(1, 21, 2):
    Max =  Score(img,side, v, d, out, color1)
    Scores.append((Max, d))
    Max =  Score(img,side, v, -d, out, color1)
    Scores.append((Max, -d))

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

'''
img = cv2.imread('2D/01.png', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('image', img)
avg, out = getAngle(img)
#cv2.imshow('out', out)
print('average angle: ', avg)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''