import os
import numpy as np
from PIL import Image
import cv2
import pickle
import io
import picamera       

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_trainer = cv2.face.LBPHFaceRecognizer_create()

#get previously saved trained model
model_trainer.read("trained_model.yml")


names = {}

#get labels from previously saved document
with open("data.pickle", "rb") as file:
    switch_names=pickle.load(file)
    names = {v:k for k,v in switch_names.items()}

print(names)


camera = picamera.PiCamera()
stream = io.BytesIO()
facesCaptured = 0 

#start a continuous loop to capture each frame from the camera
while(True):
    camera.capture(stream, use_video_port=True, format='jpeg')
    stream.seek(0)
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, 1)
    frame = cv2.blur(frame,(3, 3))
    
    #convert frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect face using CascadeClassifier
    find_face = faceCascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)    
    
    #for each face, feed the image of the face to the model and make predictions
    for(x,y,w,h) in find_face:
        gray_face = gray_img[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
        user, confidence = model_trainer.predict(cv2.resize(gray_face,(280, 280)))
        
        #ensure the model's confidence level is in the appropriate range, and then label and put a box around the recognized faces
        if confidence >= 60 and confidence<=100 :
            cv2.putText(frame, names[user], (x,y), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 1, cv2.LINE_AA)
            print("You are: "+ str(names[user]) + ", confidence: "+ str(confidence))
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), 1,)

    cv2.imshow('video', frame)
    
    #press q to stop the video
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

