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

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
camera.vflip = True
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
fps = FPS().start()

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = f.array
        frame = imutils.resize(frame, width=400)

            # detect people in the image
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8),
                                            padding=(24, 24), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
        # check to see if the frames should be displayed to screen
        if conf["show_video"]:
                # display the security feed
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
                k = cv2.waitKey(1) & 0xff
                if k == ord("q"):
                    break
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        fps.update()
        
fps.stop()
print("INFO elapsed time: {:.2f}".format(fps.elapsed()))
print("INFO approx FPS {:.2f}".format(fps.fps()))
