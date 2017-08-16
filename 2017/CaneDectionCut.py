'''input: 
% sTh: threshold for S channel,used for remove background
% rTh: threshold for R channel,used for separate yellow and greed canes
% sizeTh: threshold for size filter
% dTh: threshold for checking whether it's at the middle

output
% comd = 1, a yellow cane was detected
% comd = 0, no yellow cane was detected
'''

# This code works for Python 2.7, OpenCV 2.7.
import numpy as np
import cv2
import picamera.array
import picamera
import sys
import time 
from Phidget22.Devices.RCServo import *
from Phidget22.PhidgetException import *
from Phidget22.Phidget import *
from Phidget22.Net import *
import RPi.GPIO as GPIO
import serial

def caneDetection(rgb, chS_De, rTh):
  chR = rgb[:,:,2]
#	cv2.imshow('red channel',chR)
  [r,c] = chS_De.shape
  caneY_R = np.zeros((r,c),np.uint8)
  caneG_R = np.zeros((r,c),np.uint8)
  caneY_R[np.logical_and(chS_De>0, chR>rTh)] = 255
  caneG_R[np.logical_and(chS_De>0, chR<rTh)] = 255
  caneY = caneY_R
  caneG = caneG_R
  return caneY, caneG

def detect(rgb_raw, sTh,rTh,sizeTh,dTh,disGYTh):
  rows,cols,c = rgb_raw.shape
  #rgb=rgb_raw[:,cols/3:cols*2/3,:]
  rgb=rgb_raw
  rows,cols,c = rgb.shape
  
  M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
  rgb = cv2.warpAffine(rgb,M,(cols,rows))
  cv2.imwrite(str(time.time())+'.jpg',rgb)
  # RGB to HSV
  hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
  chS = hsv[:,:,1]   # channel S used to detect yellow canes
  
  
  # removew backgrow using Channel S
  #sTh = 0.5*255 # threshold 
  
  ########
  #chG = cv2.inRange(rgb[:,:,1], 0, 70)
  #scipy.io.savemat('a.mat',dict(green=rgb[:,:,1]))
  #cv2.imshow('green channel',chG)
  chS = cv2.inRange(chS, sTh,255)
  #cv2.imshow('yellow canes from green',chG)
  chS[rgb[:,:,1]<70] = 0
  #cv2.imwrite('yellow canes.jpg',chS)
  
  
  areaopenS = 4
  kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(areaopenS,areaopenS))
  chS_De = cv2.morphologyEx(chS, cv2.MORPH_DILATE, kernal)
  #cv2.imwrite('denoised yellow canes.jpg',chS_De)
  caneY = chS_De
  
  _,contoursY,hierarchyY = cv2.findContours(caneY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#  print('num contours Y:')
#  print(len(contoursY))

  #contourImage=np.zeros(caneY.shape)
  #cv2.drawContours(contourImage,contoursY,-1,(255,255,255),2)
  #cv2.imwrite('contoursY.jpg',contourImage) 
  
  
  lab_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
  a_channel = lab_image[:,:,1]      # a channel used to detect green canes
  #l_channel,a_channel,b_channel = cv2.split(lab_image)
  
  #a_channel = a_channel - 128
  #a_channel = cv2.inRange(a_channel, -1,128)
  a_channel = cv2.inRange(a_channel, 0,127)
  #print(np.unique(a_channel))
  #a_channel[a_channel == 255] = 1
  #a_channel[a_channel == 0] = 255
  #a_channel = cv2.inRange(a_channel, 80,255)
  a_channel[rgb[:,:,0] < 20] = 0
  #cv2.imwrite('green canes.jpg',a_channel)
  
  kernel = np.ones((2,2),np.uint8)
  a_channel = cv2.dilate(a_channel,kernel,iterations = 1)     # dilate contours in order to connnect boundary pixel
  areaopenS = 4
  kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(areaopenS,areaopenS))
  a_channel_De = cv2.morphologyEx(a_channel, cv2.MORPH_DILATE, kernal)
  #cv2.imwrite('denoised green.jpg',a_channel_De)
  caneG = a_channel_De
  

 
  _,contoursG,hierarchyG = cv2.findContours(caneG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#  print('num contours G:')
#  print(len(contoursG))
  
  #contourImage=np.zeros(caneG.shape)
  #cv2.drawContours(caneG,contoursG,-1,(255,255,255),2)
  #cv2.imwrite('contoursG.jpg',caneG) 
  
  
  #sizeTh = 30
  #sTh = 50   # threshold for checking whether it's at the middle
  
  # checke whether the green cane is at the middle and has large enough pixle number
  imgWd = caneG.shape[1]
  
  #print('image size')
  #print(caneG.shape)
  bottomCornerG=[]
  caneG_real=[]
  middleG=[]
  for i in range(len(contoursG)):
      cnts = contoursG[i]
      #print('cnts[0]')
      #print(cnts[:,0,0])
      # 0 is x, 1 is y
      bottom=np.amax(cnts[:,0,1]) #y
      top=np.amin(cnts[:,0,1])
      numG = abs(top-bottom)     # distance
      #print('numG')
      #print(numG)
      #numG = len(cnts)
      left=np.amin(cnts[:,0,0])
      shift = abs(imgWd/2-left)     # check whether the cane is at middle of the image
      #print(shift)
      #print('NumG:')
      #print(numG)
      if (numG > sizeTh) and (shift < bottom*0.0559+dTh):
          caneG_real.append(contoursG[i])     # green cane detected
          bottomCornerG.append([left,bottom])
          middleG.append((top+bottom)/2)
          
  #caneG1=np.zeros(caneY.shape)
  #cv2.drawContours(caneG1,caneG_real,-1,(255,255,255),2)
  #cv2.imwrite('contoursLeftG.jpg',caneG1)
  
  #print ('how many G')
  #print (len(caneG_real))

  # check whether the green cane should be cut       
  caneY_real=[]
  bottomCornerY=[]
  middleY=[]
  for i in range(len(contoursY)):
      cnts = contoursY[i]
      bottom=np.amax(cnts[:,0,1])
      top=np.amin(cnts[:,0,1])
      
      numY = abs(top-bottom)
      #print('Size:')
      #print(numY)
      left=np.amin(cnts[:,0,0])
      shift = abs(imgWd/2-left)
      #print ('Y shift:')
      #print shift
      if (numY > sizeTh) and (shift < bottom*0.0559+dTh):               # cane size and position information used for judegment
          caneY_real.append(contoursY[i])     # yellow cane detected
          bottomCornerY.append([left,bottom])
          middleY.append((top+bottom)/2)
      #print ('Y shift:')
      #print shift

  #caneY1=np.zeros(caneY.shape)
  #cv2.drawContours(caneY1,caneY_real,-1,(255,255,255),2)
  #cv2.imwrite('contoursLeftY.jpg',caneY1)
  
  #print ('how many Y')
  #print (len(caneY_real))
  
     
  k = 0        
  numY = len(caneY_real)       
  numG = len(caneG_real)
  
  #print('numG')
  #print(numG)
  #print('numY')
  #print(numY)
  
  for iG in range(numG):
      #centroidG = np.mean(caneG_real[iG], 0)
      #shift = shiftAllG[iG]#abs(imgWd/2-centroidG[0][0])
      
      #print('shiftG:')
      #print(shift)
      #print('centroidG:')
      #print(centroidG)
      #print(centroidG[0][0])
      #print(centroidG[0][1])
      HG = middleG[iG]
      WG = bottomCornerG[iG][0]
      
      #print('WG')
      #print(WG)
      #print('HG')
      #print(HG)
      
      if numY == 0:
          k = 1
      else:
         for iY in range(numY):
            centroidY = np.mean(caneY_real[iY], 0)
            HY = middleY[iY]
            WY = bottomCornerY[iY][0]
            
            #print('WY')
            #print(WY)
            #print('HY')
            #print(HY)
            
            disGY = abs(WG-WY)
            
            #print('disGY:')
            #print(disGY)
            
            if (HG > HY) and (disGY < disGYTh):     # check whether the green cane is in front of yellow canes
                k = k+1
          
      
  if k >= 1:
      comd = True
  else:
      comd = False

  return comd


def detectAndCutCane(camera):
  sTh = 0.5*255
  rTh = 70
  sizeTh = 20
  dTh = 200   # threshold for checking whether it's at the middle
  disGYTh = 110
  comd = False

  #camera.capture('start.jpg')
  stream=picamera.array.PiRGBArray(camera)
  camera.capture(stream, format='bgr')
  rgb_raw = stream.array
  start = time.clock()
  rgb_raw=cv2.imread('1500402125.38.jpg')
  # detect the cane
  comd=detect(rgb_raw,sTh,rTh,sizeTh,dTh,disGYTh)
  elapsed = (time.clock() - start)
  print('comd = %s' %comd)
  print('time elapsed = %s' %elapsed)
  if False:
    i=50
    t0=time.time()
    stop=False
    noCaneReached=False
    while 1:
	# every time move the linear slider a little bit and test whether the switch is triggered or not.
      ch0.setTargetPosition(i)
      ch0.setEngaged(1)
      i=i+5
        
      
      time.sleep(0.5)
	  # get the elapsed time
      t1=time.time()-t0
      if False:
        print('Don\'t cut this cane.\n')
        stop=True
        break
      else:
	  # if the switch is triggered, stop the linear acuator
        if GPIO.input(switch):
          print('Cane reached.\n')
      #        tp=ch0.getTargetPosition()
      #        print(tp)
      #        ch0.setTargetPosition(i)
          ch0.setEngaged(0)
          break
        if t1>8:
		# timeout, no cane is reached.
          print("No Cane reached\n")
          noCaneReached=True
          break

      
      #time.sleep(1)
	  # if the switch is triggered and the cane is weak cane, cut it.
    if not stop and not noCaneReached:
      print("Cut cane\n")
    #time.sleep(0.5)
      cutter.ChangeDutyCycle(2)
      time.sleep(1.7)
	 # retract the arm
    print("Retract arm...")
    ch0.setTargetPosition(45)
    ch0.setEngaged(1)
    if not stop and not noCaneReached:
      cutter.ChangeDutyCycle(30)
	# if time is enough, move the cutter twice.
    if t1<1.7:
      time.sleep(1.7+2)
    else:
      if t1<=5.1:
        time.sleep(t1-1.7+2)
      else:
        time.sleep(1.7)
        if not stop and not noCaneReached:
          cutter.ChangeDutyCycle(2)
        time.sleep(1.7)
        if not stop and not noCaneReached:
          cutter.ChangeDutyCycle(30)
        time.sleep(t1-5.1+2)
    
  return comd
try:
    ch0 = RCServo()
    
except RuntimeError as e:
    print("Runtime Exception %s" % e.details)
    print("Press Enter to Exit...\n")
    readin = sys.stdin.read(1)
    exit(1)

def RCServoAttached(e):
    try:
        attached = e

        print("\nAttach Event Detected (Information Below)")
        print("===========================================")
        print("Library Version: %s" % attached.getLibraryVersion())
        print("Serial Number: %d" % attached.getDeviceSerialNumber())
        print("Channel: %d" % attached.getChannel())
        print("Channel Class: %s" % attached.getChannelClass())
        print("Channel Name: %s" % attached.getChannelName())
        print("Device ID: %d" % attached.getDeviceID())
        print("Device Version: %d" % attached.getDeviceVersion())
        print("Device Name: %s" % attached.getDeviceName())
        print("Device Class: %d" % attached.getDeviceClass())
        print("\n")

    except PhidgetException as e:
        print("Phidget Exception %i: %s" % (e.code, e.details))
        print("Press Enter to Exit...\n")
        readin = sys.stdin.read(1)
        exit(1)   
    
def RCServoDetached(e):
    detached = e
    try:
        print("\nDetach event on Port %d Channel %d" % (detached.getHubPort(), detached.getChannel()))
    except PhidgetException as e:
        print("Phidget Exception %i: %s" % (e.code, e.details))
        print("Press Enter to Exit...\n")
        readin = sys.stdin.read(1)
        exit(1)   

def ErrorEvent(e, eCode, description):
    print("Error %i : %s" % (eCode, description))

def PositionChangeHandler(e, position):
    print("Position: %f" % position)

try:
    ch0.setOnAttachHandler(RCServoAttached)
    ch0.setOnDetachHandler(RCServoDetached)
    ch0.setOnErrorHandler(ErrorEvent)

    ch0.setOnPositionChangeHandler(PositionChangeHandler)

    print("Waiting for the Phidget RCServo Object to be attached...")
    ch0.openWaitForAttachment(5000)
except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    print("Press Enter to Exit...\n")
    readin = sys.stdin.read(1)
    exit(1)

isPortOpen=False
#try:
#  port = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=0.1)
#  isPortOpen=True
#except:
#  print('Can\'t open port.\n')
#  isPortOpen=False


# open picamere
camera = picamera.PiCamera()

# GPIO pin number for the switch under the cutter
switch=11
# GPIO pin number for the cutter servo motor
cutter_pin=12
GPIO.setmode(GPIO.BOARD)

# configure the switch pin as input
GPIO.setup(switch, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(cutter_pin, GPIO.OUT)

# pin 16 and 18 are used for communication between the robot controller and pi.
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(18, GPIO.OUT)

# servo motor for the cutter is controlled using pwm
cutter=GPIO.PWM(cutter_pin, 100)
cutter.start(30);

# linear acuator controller setup
ch0.setMinPosition(0)
ch0.setMaxPosition(180)
ch0.setDataInterval(ch0.getMinDataInterval())

start_time=time.time()

# store how many canes are cut
iCut=1
#if isPortOpen:
#  port.write('G')
print('start')
GPIO.output(18, GPIO.HIGH)
while True:
  try:
    #if True:
    #if isPortOpen:
      #ch = port.readline()
      
      #print ('%s' %ch)
    #print(GPIO.input(16))
    #if GPIO.input(16):
    if True:
      GPIO.output(18, GPIO.LOW)
      #if True:
      time.sleep(1)
      t0=time.time()
      # detect and cut the cane
      res=detectAndCutCane(camera)
      t1=time.time()-t0
        
      print('%d detect, use time = %f, result =%d' % (iCut, t1, res))
        
      iCut=iCut+1
      # Send a high signal to the robot to state the cutting process ends
      GPIO.output(18, GPIO.HIGH)
      #time.sleep(0.001)
  except:
    #port.close()
    camera.close()
	# pull the line low
    GPIO.output(18,GPIO.LOW)
    
    sys.exit(0)


try:
    ch0.close()

except PhidgetException as e:
    print("Phidget Exception %i: %s" % (e.code, e.details))
    print("Press Enter to Exit...\n")
    readin = sys.stdin.read(1)
    exit(1) 
print("Closed RCServo device")
exit(0)
                     




