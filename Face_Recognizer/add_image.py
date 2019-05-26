import cv2
import os

def take():
    print("Get ready")
    camera = cv2.VideoCapture(1)
    x = os.listdir("Test-data")
    l = len(x) + 1
    input("press Enter to capture")
    return_value, image = camera.read()
    path = "Test-data/test"+ str(l) +".jpg"
    cv2.imwrite(path,image)
    img = cv2.imread(path)
    cv2.imshow("Image",img)
    cv2.waitKey(100)
