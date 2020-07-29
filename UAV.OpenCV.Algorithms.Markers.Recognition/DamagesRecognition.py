# -*- coding: utf-8 -*-
"""
Building Damages Markers Recognition - ERL Emergency 2020
Created on Fri Mar  6 12:01:13 2020

@author: Marta Kwiatkowska
"""
import numpy as np
import cv2 as cv

def DetectBuildingDamages(img):
    mask = DetectRedColor(img)
    DrawCountorus(mask, img)
    
    if cv.countNonZero(mask):
        print ("Detected damage(s).")
    else:
        print ("No damages detected.")

def DetectRedColor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower_red = np.array([0,120,70])
    # upper_red = np.array([5,255,255])
    lower_red = np.array([161, 155, 8])
    upper_red = np.array([179, 255, 255])
    mask = cv.inRange(hsv, lower_red, upper_red)
    return mask

def DrawCountorus(mask, img):
    image,cnts,hie = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, cnts, -1, (0,255,255), thickness=1) 
    # for c in cnts:
    #     x,y,w,h = cv.boundingRect(c)
    #     cv.putText(img, 'Building damage', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
if __name__ == "__main__":
    
 # # code for the image
    
 #    source_path = "D:\\repos\\UAV.OpenCV.Algorithms\\UAV.OpenCV.Algorithms.Markers.Recognition\\Images\\BuildingDamages.png"
 #    img = cv.imread(source_path) 

 #    DetectBuildingDamages(img)    
    
 #    cv.imshow('image',img)
 #    cv.waitKey(0)
 #    cv.destroyAllWindows()    
    
# code for the video capture
    
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        DetectBuildingDamages(frame) 
    
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()