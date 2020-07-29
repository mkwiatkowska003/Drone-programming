# -*- coding: utf-8 -*-
"""
Missing Worker Detection - ERL Emergency 2020
Created on Fri Mar  6 12:01:13 2020

@author: Marta Kwiatkowska
"""

import numpy as np
import cv2 as cv
from imutils.object_detection import non_max_suppression

def DetectMissingWorker(img):    
    DetectFullBody(img)
    is_face_detected = DetectFace(img)
    mask = DetectOrangeColor(img)
    DrawCountorus(mask, img)
    
    if cv.countNonZero(mask) & is_face_detected :
        print ("Detected missing worker.")
    else:
        print ("No workers detected.")

def DetectOrangeColor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_orange = np.array([0, 200, 200])
    upper_orange = np.array([25, 255, 255])
    # lower_orange = np.array([10, 100, 20])
    # upper_orange = np.array([25, 255, 255])
    mask = cv.inRange(hsv, lower_orange, upper_orange)
    return mask

def DrawCountorus(mask, img):
    image,cnts,hie = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, cnts, -1, (0,255,255), thickness=1) 
    #x,y,w,h = cv.boundingRect(cnts[len(cnts)-1])
    #cv.putText(img, 'Missing worker', (x+100, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
def DetectFace(img):
    detector = cv.CascadeClassifier("C:\\Users\\marta\\PycharmProjects\\OpenCV.Algorithms\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (i, (x, y, w, h)) in enumerate(rects):
    	cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), thickness=1)
    	cv.putText(img, "Face #{}".format(i + 1), (x, y - 10),
    		cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        
    if rects is not None:
        return True
    else:
        return False
        
def DetectFullBody(img):
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    
    (rect, weight) = hog.detectMultiScale(img, winStride=(4, 4), 	padding=(8, 8), scale=1.05)
        
    rect = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rect])
    pick = non_max_suppression(rect, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
    	cv.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 1)
        
if __name__ == "__main__":
    
# code for the image
    
    source_path = "D:\\repos\\UAV.OpenCV.Algorithms\\UAV.OpenCV.Algorithms.Missing.Worker\\Images\\Worker.png"
    img = cv.imread(source_path) 

    DetectMissingWorker(img) 
    
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
