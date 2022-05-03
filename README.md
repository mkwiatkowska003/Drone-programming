# Intoduction
The project contains a set of python algorithms for a simple drone control using MAVSDK library. Solution has been prepared for an international robotics competition with the use of drones, and on this basis the nomenclature of source files and some variables used in the programs were adopted. All of these algorithms can be run as a standalone application in reference to the appropriate directory. Images in specified directories could be replaced with any pictures you want to use in the process. 

# Requirements
All programs requires installed Python 3.6 with the MAVSDK tools, a real drone connected to your computer and configured correctly. Installing QGroundControl could be helpful for calibration and real-time telemetry control.

# Tasks and missions
The following list contains the solutions for some basic drone's tasks and missions: <br/>
<i>waypoints.py</i> - create and run a simple mission with some waypoints defined as latidude and longitude angles. <br/>
<i>UAV.OpenCV.Algorithms.RealsenceDepthCamera</i> (all files) - use RealSense Depth camera and apply real-time object detection algorithms using SSD neural network and MobileNet.<br/>
<i>UAV.OpenCV.Algorithms.ObstacleAvoidance</i> (all files) - example of program for a simple obstacle avoidance using a standard and a RealSense Depth cameras.<br/>
<i>UAV.OpenCV.Algorithms.Missing.Worker</i> (all files) - example of human shapes detection using opencv and neural network. <br/>
<i>UAV.OpenCV.Algorithms.Markers.Recognition</i> (all files) - example of colors detection using opencv. <br/>
<i>UAV.OpenCV.Algorithms.Self.Camera/main.py</i> - a helpful program to read the video image from your webcam. <br/>
