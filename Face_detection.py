import io
import picamera
import cv2
import numpy

#create a memory stream so photos not have to be saved within a file.
stream = io.BytesIO()

#getting the picture(lower res. is faster)
#specify other paramters needed also ie. image rotation
with picamera.PiCamera() as camera:
	camera.resolution = (320, 240)
	camera.capture(stream, format='jpeg')
	camera.rotation = 0
	
#convert the picture into numpy array
buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

#now create a opencv image
image = cv2.imdecode(buff, 1)

#load a cascade file for detetcting faces
face_cascade = cv2.CascadeClassifier('/home/pi/Documents/Pic_Vid/haarcascade_frontalface_default.xml')

#convert to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#look for faces in the image using the loaded cascade file
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

print("Found "+str(len(faces))+" face(s)")

#draw a rectangle around every face
for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

#save the resulting image
cv2.imwrite('/home/pi/Documents/Pic_Vid/result.jpg',image)
