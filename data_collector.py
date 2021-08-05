import io
import time
import os
from PIL import Image
import cv2
import pickle
from picamera.array import PiRGBArray
from picamera import PiCamera


#takes user's name and creates a subdirectory in data to save the images
#collected for training
first_name = input('Enter your first name: ')
last_name = input('Enter your last name: ')
folder_name = 'data/'+first_name.title()+'_'+last_name.title()
os.mkdir(folder_name)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#take webcam video, need to change for pi
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))
data_img_num = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    #initialize variable image as a numpy array representing the frame captured
    image = frame.array

    #convert to gray
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #identify faces in frame using faceCascade file
    find_face = faceCascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    cv2.imshow("Frame", image)

    # Wait for key
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    for(x,y,w,h) in find_face:
        gray_face = gray_img[y:y+h, x:x+w]
        color_face = image[y:y+h, x:x+w]
        cv2.putText(image, "Storing Data...", (x,y), cv2.FONT_HERSHEY_SIMPLEX,.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.imwrite(folder_name + "/" + str(data_img_num)+".JPG", image)
        data_img_num +=1

    if data_img_num == 40:
            break
   

    
    
    


