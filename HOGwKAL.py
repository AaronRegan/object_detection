# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from non_max_suppression import non_max_suppression
import numpy as np
from imutils.video import FPS
import argparse
import warnings
import datetime
import imutils
import time
import cv2
import json

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
        help="path to the JSON configuration file")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
camera.vflip = True
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

meas = []
pred = []
mp = np.array((2, 1), np.float32)  # measurement
tp = np.zeros((2, 1), np.float32)  # tracked / prediction

totalHOG = 0

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

def onPed(x, y):
    global mp, meas
    mp = np.array([[np.float32(x)], [np.float32(y)]])
    meas.append((x, y))

def updateKalman(mp):
    global pred, tp
    kalman.correct(mp)
    tp = kalman.predict()
    pred.append((int(tp[0]), int(tp[1])))

def paint(tp, xA, yA, xB, yB):
    global frame, pred
    cv2.circle(frame, ((tp[0]), (tp[1])), 3, (0, 0, 255), -1)
    cv2.rectangle(frame, ((tp[0]) - ((xB - xA) / 2), (tp[1]) + (yB - yA) / 2),
                  (((tp[0]) + ((xB - xA) / 2)), ((tp[1]) - (yB - yA) / 2)), (0, 0, 255), 2)
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
fps = FPS().start()

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image and initialize
        # the timestamp and occupied/unoccupied text
        frame = f.array
        timestamp = datetime.datetime.now()

        frame = imutils.resize(frame, width=250)
        e1 = cv2.getTickCount()
            # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(24, 24), scale=1.05)
        e2 = cv2.getTickCount()
        timeHOG = (e2-e1)/cv2.getTickFrequency()
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            centerX = (xB + xA) / 2
            centerY = (yB + yA) / 2
            onPed(centerX, centerY)
            updateKalman(mp)
            paint(tp, xA, yA, xB, yB)
        # check to see if the frames should be displayed to screen
        output = imutils.resize(frame, width=600)
        print_HOG_time = "[INFO] HOG RUN TIME: "+ str(timeHOG)
        cv2.putText(output, print_HOG_time, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Frame", output)
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
            break
     
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        fps.update()
        totalHOG= timeHOG+totalHOG
totalAvgHOG = totalHOG/fps._numFrames

fps.stop()
print("INFO elapsed time: {:.2f}".format(fps.elapsed()))
print("INFO approx FPS {:.2f}".format(fps.fps()))
print("INFO Average time to complete HOG: {:.2f} Seconds".format(totalAvgHOG))
