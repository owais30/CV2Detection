import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import extract_names as ext
import loading_test_numbers as lts
import take_pictures as tp
import add_image as ai




print ("Enter 1 if you are a new person and you want to add new data")
inp = input()
if inp == "1":
    tp.take()
print ("Enter 5 if you want to test your image")
inpt = input()
if inpt == "5":
    ai.take()

subjects = ext.names()
print(subjects)


def detect_face(img):
    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    if (len(faces) == 0):
        return None, None

    
    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]



def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)
 
    faces = []

    labels = []

    for dir_name in dirs:
 

        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
 

        subject_dir_path = data_folder_path + "/" + dir_name
 

        subject_images_names = os.listdir(subject_dir_path)
 

        for image_name in subject_images_names:
 

            if image_name.startswith("."):
                continue;
            if image_name.startswith("_"):
                continue;
 

            image_path = subject_dir_path + "/" + image_name
 

            image = cv2.imread(image_path)
 
 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
 

            face, rect = detect_face(image)
 

            if face is not None:

                faces.append(face)

                labels.append(label)
 
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
 
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
 

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))




 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
 (x, y, w, h) = rect
 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 

def draw_text(img, text, x, y):
 cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):

    img = test_img.copy()

    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        label= face_recognizer.predict(gray[y:y+w, x:x+h])
        label_text = subjects[label[0]]
        draw_text(img,label_text,x, y-5)
 

 
    return img


print("Predicting images...")

print(lts.test_data())
for im in lts.test_data():
    test_img = cv2.imread("test-data/"+im)
    predict_img = predict(test_img)
    cv2.imshow("Test_Image_" + im , predict_img)
k = cv2.waitKey(30) & 0xff
if k == 27:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

vinp = input("Enter YES to detect faces from video: ")

if vinp == "YES":
    cap = cv2.VideoCapture(1)
    while 1:
        ret, img = cap.read()
        predict_img = predict(img)
        cv2.imshow("Test_Image_Cap", predict_img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()   
    
cv2.waitKey(0)
cv2.destroyAllWindows()







