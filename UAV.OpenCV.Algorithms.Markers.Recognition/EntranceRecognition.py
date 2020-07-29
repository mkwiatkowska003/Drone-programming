# -*- coding: utf-8 -*-
"""
Entrance Markers Recognition - ERL Emergency 2020
Created on Fri Mar  6 12:01:13 2020

@author: Marta Kwiatkowska
"""

import numpy as np
import cv2 as cv

def DetectBlockedEntrances(img):
    mask = DetectBlueColor(img)
    DrawCountorus(mask, img, 'Blocked entrance')
    
    if cv.countNonZero(mask):
        print ("Detected blocked entrance(s).")
    # else :
    #     print ("There is no blocked entrance.")
        
def DetectUnblockedEntrances(img):
    mask = DetectGreenColor(img)
    DrawCountorus(mask, img, 'Unblocked entrance')
    
    if cv.countNonZero(mask):
        print ("Detected unblocked entrance(s).")
    # else :
    #     print ("There is no unblocked entrance.")

def DetectBlueColor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    return mask

def DetectGreenColor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # lower_green = np.array([65,150,60])
    # upper_green = np.array([70,255,255])
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    return mask

def DrawCountorus(mask, img, text):
    image,cnts,hie = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, cnts, -1, (0,255,255), thickness=1) 
    # for c in cnts:
    #     x,y,w,h = cv.boundingRect(c)
    #     cv.putText(img, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

if __name__ == "__main__":
    
# code for the image
    
    source_path = "D:\\repos\\UAV.OpenCV.Algorithms\\UAV.OpenCV.Algorithms.Markers.Recognition\\Images\\Entrances.png"
    img = cv.imread(source_path)

    DetectBlockedEntrances(img) 
    DetectUnblockedEntrances(img)       
    
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()    
    
# code for the video capture
    
#     cap = cv.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         DetectBlockedEntrances(frame)
#         #DetectUnblockedEntrances(frame)
    
#         cv.imshow('frame', frame)
#         if cv.waitKey(1) == ord('q'):
#             break

# cap.release()
# cv.destroyAllWindows()
    