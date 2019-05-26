import cv2
import os

def take():
    print("Get ready")
    camera = cv2.VideoCapture(1)
    x = os.listdir("Training-data")
    l = len(x) + 1
    i = 0
    directory = "Training-data/s" + str(l)
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = input("Enter your name")
    person = directory + "/_" + name
    os.makedirs(person)

    while i < 15:
        input("press Enter to capture")
        return_value, image = camera.read()
        path = directory + "/"+ str(i) +".jpg"
        cv2.imwrite(path,image)
        img = cv2.imread(path)
        cv2.imshow("Image",img)
        cv2.waitKey(100)
        i = i + 1
       
        
        
        
