# import the necessary packages
from imutils.video.pivideostream import PiVideoStream
from non_max_suppression import non_max_suppression
from picamera.array import PiRGBArray
from imutils.video import FPS
from picamera import PiCamera
import numpy as np
import datetime
import argparse
import warnings
import imutils
import time
import cv2
import json

#get path to JSON file
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,help="path to the JSON configuration file")
args = vars(ap.parse_args())
#load JSON file from path
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None
#initliase our matrices for storing Kalman values   
meas = []   #all collected measured values
pred = []   #all collected predicted values
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction
#intialzing variables for storing clock times
totalHOG = 0
totalKAL = 0
#assign the descriptor to its variable & call the people Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#initialising Kalman filter and assign the measurement,transition and noise values
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

def onPed(x, y):#on detection of pedestrian we pass in the x,y co-ordinates
    global mp, meas
    mp = np.array([[np.float32(x)], [np.float32(y)]])
    meas.append((x, y))

def updateKalman(mp):   #updateKalman as measured is passed into kalman 
    global pred, tp     #the predicted kalman is then returned , the rectangle is drawn
    kalman.correct(mp)  #aroundthis  point
    tp = kalman.predict()
    pred.append((int(tp[0]), int(tp[1])))

def paint(tp, xA, yA, xB, yB):#draw rectangle for predicted state
    global frame, pred
    cv2.circle(frame, ((tp[0]), (tp[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(frame, ((tp[0]) - ((xB - xA) / 2), (tp[1]) + (yB - yA) / 2),
                  (((tp[0]) + ((xB - xA) / 2)), ((tp[1]) - (yB - yA) / 2)), (0, 0, 255), 2)
 
#start the stream , wait for 2 seconds for the camera to warm up
vs = PiVideoStream().start()
print("[INFO] warming up...")
time.sleep(2.0)
fps = FPS().start()
avg = None

#perform this loop until the user exits
while(True):
        #grab the frame from the top of the queue
        frame = vs.read()
        #resize to 300
        frame = imutils.resize(frame, width=300)
        frame = cv2.flip(frame,0)
        # detect people in the image
        e1 = cv2.getTickCount()
         #run detection with HOG over frame (specified parameters best tested
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(16, 16), scale=1.05)
        e2 = cv2.getTickCount()
        timeHOG = (e2-e1)/cv2.getTickFrequency()
        #the HOG returns rectangles, for every rectangle
        #carry out non_max_suppresion
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        #loop over every returned co-ordinate and draw the bounding box
        f1 = cv2.getTickCount()
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            centerX = (xB + xA) / 2
            centerY = (yB + yA) / 2
            #collect the center value and pass this into the kalman mesaured function
            onPed(centerX, centerY)
            #updatekalmanwith the measured point
            updateKalman(mp)
            #the returned predicted co-ordinate is passed into the paint function
            paint(tp, xA, yA, xB, yB)
        f2 = cv2.getTickCount()
        timeKAL = (f2-f1)/cv2.getTickFrequency()
        # check to see if the frames should be displayed to screen
        output = imutils.resize(frame, width=600)
        #resize frame to make it easier to view
        print_HOG_time = "[INFO] HOG RUN TIME: "+ str(timeHOG)
        cv2.putText(output, print_HOG_time, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.namedWindow("HOG", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("HOG",cv2.WND_PROP_FULLSCREEN,1)
        cv2.imshow("HOG", output)
        k = cv2.waitKey(1) & 0xff
        #if Q is pressed quit the loop of frames
        if k == ord("q"):
            break
        fps.update()
        print ("{}".format(timeHOG)
        totalKAL= timeKAL+totalKAL
        totalHOG= timeHOG+totalHOG
#print all stored Data for analysis
totalAvgHOG = totalHOG/fps._numFrames
totalAvgKAL = totalKAL/fps._numFrames
fps.stop()
print("INFO elapsed time: {:.2f}".format(fps.elapsed()))
print("INFO approx FPS {:.2f}".format(fps.fps()))
print("INFO Average time to complete HOG: {:.2f} Seconds".format(totalAvgHOG))
print("INFO Average time to complete NMS: {:.2f} Seconds".format(totalAvgKAL))
cv2.destroyAllWindows()
vs.stop()
