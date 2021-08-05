import os
from PIL import Image
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(BASE_DIR, "data")

model_trainer = cv2.face.LBPHFaceRecognizer_create()
id = 0
names = {}
y_cord= []
x_cord = []

#Find all images in the data folder, convert to grayscale
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("JPG"):
            path = os.path.join(root,file)
            name = os.path.basename(os.path.dirname(path)).title()
            if not name in names:
                names[name] = id
                id += 1
            new_user = names[name]
            pimage = Image.open(path).convert("L")
            altered_img = pimage.resize((550,550), Image.ANTIALIAS)

            np_img = np.array(altered_img, "uint8")
            find_face = faceCascade.detectMultiScale(np_img, scaleFactor=1.1, minNeighbors=5)

            #add face images and corresponding labels to lists that will later be used to train model
            for(x,y,w,h) in find_face:
                face = np_img[y:y+h, x:x+w]
                x_cord.append(cv2.resize(face,(280,280)))
                y_cord.append(new_user)

#save labels to file
with open("data.pickle", "wb") as file:
    pickle.dump(names, file)

#train model with the created list
model_trainer.train(x_cord, np.array(y_cord))
model_trainer.save("trained_model.yml")


def testAccuracy(X, y):
    model = cv2.face.LBPHFaceRecognizer_create()
    #model = cv2.face.EigenFaceRecognizer_create()
    #model = cv2.face.FisherFaceRecognizer_create()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.train(X_train, np.array(y_train))
    y_pred = []
    for x in X_test: 
        user, confidence = model.predict(x)
        y_pred.append(user)
    print(y_pred)
    print(y_test)
    print(classification_report(y_test, y_pred, target_names = names))
    print(confusion_matrix(y_test, y_pred))
    
    labels = []
    for key in names.keys():
        labels.append(key)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45,ha="left", va="center",rotation_mode="anchor")
    
    plt.yticks(tick_marks, labels)

    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.show()

testAccuracy(x_cord, y_cord)