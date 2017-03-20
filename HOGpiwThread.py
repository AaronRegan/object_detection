# import the necessary packages
from non_max_suppression import non_max_suppression
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.video import FPS
from imutils.video.pivideostream import PiVideoStream
import numpy as np
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

totalHOG = 0
totalResize = 0

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
vs = PiVideoStream().start()
time.sleep(conf["camera_warmup_time"])
avg = None
fps = FPS().start()

# capture frames from the camera
while fps._numFrames < conf["number_frames"]:
        f1 = cv2.getTickCount()
        # grab the raw NumPy array representing the image and initialize
        # the timestamp and occupied/unoccupied text
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = cv2.flip(frame,0)
        f2 = cv2.getTickCount()

        timeResize = (f2-f1)/cv2.getTickFrequency()
                
        e1 = cv2.getTickCount()
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(16, 16),
                                            padding=(16, 16), scale=1.05)
        e2 = cv2.getTickCount()
        timeHOG = (e2-e1)/cv2.getTickFrequency()
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        f1 = cv2.getTickCount()
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
                break
        fps.update()
        print(timeHOG)
        totalHOG= timeHOG+totalHOG
        totalResize= timeResize+totalResize
totalAvgHOG = totalHOG/fps._numFrames
totalAvgResize = totalResize/fps._numFrames
fps.stop()
print("INFO elapsed time: {:.2f}".format(fps.elapsed()))
print("INFO approx FPS {:.2f}".format(fps.fps()))
print("INFO Average time to complete HOG: {:.2f} Seconds".format(totalAvgHOG))
print("INFO Average time to complete NMS: {:.2f} Seconds".format(totalAvgResize))
cv2.destroyAllWindows()
vs.stop()
