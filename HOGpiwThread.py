# import the necessary packages
from imutils.video.pivideostream import PiVideoStream
from non_max_suppression import non_max_suppression
from picamera.array import PiRGBArray
from imutils.video import FPS
from picamera import PiCamera
import numpy as np
import argparse
import warnings
import datetime
import imutils
import time
import cv2
import json

#get path to JSON file
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
        help="path to the JSON configuration file")
args = vars(ap.parse_args())
#load JSON file from path
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None
#intialzing variables for storing clock times
totalHOG = 0
totalResize = 0
#assign the descriptor to its variable & call the people Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#start camera stream , first allow camera to warm up , JSON set to 2.5 seconds
print("[INFO] warming up...")
vs = PiVideoStream().start()
time.sleep(conf["camera_warmup_time"])
avg = None
#Start FPS tracker
fps = FPS().start()

#run this loop until user exits
while(True):
        #start clock for timin resize
        f1 = cv2.getTickCount()
        #read incoming frame from queue and resize and flip
        frame = vs.read()
        #frame size 300 allows for maximum classification perfromance with
        #fastest Frame rate
        frame = imutils.resize(frame, width=300)
        frame = cv2.flip(frame,0)
        f2 = cv2.getTickCount()
        
        timeResize = (f2-f1)/cv2.getTickFrequency()
                
        e1 = cv2.getTickCount()
        #run detection with HOG over frame (specified parameters best tested
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(16, 16), scale=1.05)
        e2 = cv2.getTickCount()
        timeHOG = (e2-e1)/cv2.getTickFrequency()
        #the HOG returns rectangles, for every rectangle
        #carry out non_max_suppresion
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        f1 = cv2.getTickCount()
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        #loop over every returned co-ordinate and draw the bounding box
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            #resize frame to make it easier to view
        frame = imutils.resize(frame, width=600)    
        print_HOG_time = "[INFO] HOG RUN TIME: "+ str(timeHOG)
        cv2.putText(frame, print_HOG_time, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        #make window full screen
        cv2.namedWindow("HOG", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("HOG",cv2.WND_PROP_FULLSCREEN,1)
        cv2.imshow("HOG", frame)
        #if Q is pressed quit the loop of frames
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
                break
        fps.update()
        print(timeHOG)
        totalHOG= timeHOG+totalHOG
        totalResize= timeResize+totalResize
#print all stored Data for analysis
totalAvgHOG = totalHOG/fps._numFrames
totalAvgResize = totalResize/fps._numFrames
fps.stop()
print("INFO elapsed time: {:.2f}".format(fps.elapsed()))
print("INFO approx FPS {:.2f}".format(fps.fps()))
print("INFO Average time to complete HOG: {:.2f} Seconds".format(totalAvgHOG))
print("INFO Average time to complete NMS: {:.2f} Seconds".format(totalAvgResize))
cv2.destroyAllWindows()
vs.stop()
