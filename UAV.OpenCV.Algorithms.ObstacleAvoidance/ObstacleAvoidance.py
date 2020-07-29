# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:42:22 2020

@author: Marta Kwiatkowska
"""

import numpy as np
import cv2


Detected_Points = [0,0,0]


def DetectObstacle(frame, lower_color_value, upper_color_value, color_name, color_number, shapes_bgr_values):  
    
    global PointCount
        
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color_value, upper_color_value)
    
    m_filtered = cv2.medianBlur(mask, 5)
    
    _, contours, hierarchy = cv2.findContours(m_filtered,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
        cv2.drawContours(frame, [c] ,0,shapes_bgr_values,1)
    
    if len(contours) != 0:     
        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        # shrink area
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # get center point
        x = int(sum([point[0] for point in box])/4)
        y = int(sum([point[1] for point in box])/4)

        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a rectangle
        cv2.drawContours(frame, [box], 0, shapes_bgr_values, 1)
        cv2.drawContours(mask, [box], 0, shapes_bgr_values, 1)

        Detected_Points[color_number] = 1
        point = (x,y)
    
    else:
        Detected_Points[color_number] = 0
        point = (0,0)
        
    return frame, m_filtered, point
    
    
def CheckPoints(Detected_Points, blue_point, red_point, yellow_point):
    
    if any(p == 0 for p in Detected_Points):
        if Detected_Points[0] == 0:
            if red_point[1] > yellow_point[1] > 0:
                return 1
            else:
                return 0
        elif Detected_Points[1] == 0:
            if blue_point[1] > yellow_point[1] > 0 :
                return 1
            else:
                return 0
        elif Detected_Points[2] == 0:
            if blue_point[1] > red_point[1] > 0:
                return 1
            else:
                return 0
    else:
        if blue_point[1] > red_point[1] > yellow_point[1]:
            return 1
        else:
            return 0         
        
def RescaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
            
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(True):
       ret, frame = cap.read()
       
       rescaled = RescaleFrame(frame, percent=50)
       
       # blue obstacle's marker
       lower_blue = np.array([38, 150, 60])
       upper_blue = np.array([121, 255, 255])
       rescaled, blue_mask, blue_point = DetectObstacle(frame, lower_blue, upper_blue, "Blue", 0, (255, 0, 0))       
       
       # red obstacle's marker
       lower_red = np.array([0,50,50])
       upper_red = np.array([10,255,255])
       rescaled, red_mask, red_point  = DetectObstacle(rescaled, lower_red, upper_red, "Red", 1, (0, 0, 255))
       
       # yellow obstacle's marker
       lower_yellow = np.array([20, 100, 100])
       upper_yellow = np.array([30, 255, 255])
       rescaled, yellow_mask, yellow_point = DetectObstacle(rescaled, lower_yellow, upper_yellow, "Yellow", 2, (0, 255, 255))
                     
       print(f"Blue: {blue_point}, Red: {red_point}, Yellow: {yellow_point} | Points: {Detected_Points}")
       
       cv2.imshow('frame',rescaled)   
       
       combined = blue_mask | red_mask | yellow_mask
       cv2.imshow('mask', combined)      
       
       IsObstacleDetected = CheckPoints(Detected_Points, blue_point, red_point, yellow_point)
       
       if IsObstacleDetected == 1:
           print("Obstacle detected")
       else:
           print("Clear to fly")
           
       if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()