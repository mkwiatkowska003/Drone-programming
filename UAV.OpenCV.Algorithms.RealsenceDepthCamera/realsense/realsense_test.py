import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import cv2
import open3d
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)

vis = open3d.visualization.Visualizer()
vis.create_window('PCD', width=1280, height=720)
pointcloud = open3d.geometry.PointCloud()
geom_added = False

while True:
    dt0 = datetime.now()
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    profile = frames.get_profile()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    img_depth = open3d.geometry.Image(depth_image)
    img_color = open3d.geometry.Image(color_image)

    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy,
                                                      intrinsics.ppx, intrinsics.ppy)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pointcloud.points = pcd.points
    pointcloud.colors = pcd.colors

    if geom_added == False:
        vis.add_geometry(pointcloud)
        geom_added = True

    vis.update_geometry()
    vis.reset_view_point(False)
    vis.poll_events()
    vis.update_renderer()

    cv2.imshow('bgr', color_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
vis.destroy_window()
del vis
