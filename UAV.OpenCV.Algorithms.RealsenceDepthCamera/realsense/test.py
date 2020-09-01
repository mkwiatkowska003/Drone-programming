from datetime import datetime
import pyrealsense2 as rs
import numpy as np
import open3d

print("Load a ply point cloud, print it, and render it")
pcd = open3d.io.read_point_cloud("pointcloud.ply")
print(pcd)
print(np.asarray(pcd.points))
open3d.visualization.draw_geometries([pcd])