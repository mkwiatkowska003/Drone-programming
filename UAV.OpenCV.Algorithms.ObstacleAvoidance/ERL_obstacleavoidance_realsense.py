import pyrealsense2 as rs
import numpy as np
import cv2
import os
import ObstacleAvoidance as obs_avoidance


# def combine_boxes(top, center, bottom):
#     (x1, y1, w1, h1) = top.astype("int") # top box
#     (x2, y2, w2, h2) = center.astype("int") # center box
#     (x3, y3, w3, h3) = bottom.astype("int") # bottom box
#
#     x_combined = x3
#     y_combined = y3
#     w_combined = x1 + w1
#     h_combined = y1 + h1
#
#     return x_combined, y_combined, w_combined, h_combined


def get_depth_as_distance(box, curr_frame, depth_image, color_image, depth_intrin):
    (x, y, w, h) = box.astype("int")
    if curr_frame >= 30 and curr_frame % 30 == 0:
        roi_depth_image = depth_image[y:y + h, x:x + w]
        roi_color_image = color_image[y:y + h, x:x + w]

        os.system('mkdir %d' % curr_frame)
        cv2.imwrite('%d/depth.jpg' %
                    curr_frame, roi_depth_image)
        print("save depth image")
        cv2.imwrite('%d/color.jpg' %
                    curr_frame, roi_color_image)
        print("save color image")
        print("the mid position depth is:", depth_frame.get_distance(
            int(x + w / 2), int(y + h / 2)))

        # write the depth data in a depth.txt
        with open('%d/depth.csv' % curr_frame, 'w') as f:
            cols = list(range(x, x + w))
            rows = list(range(y, y + h))
            for i in rows:
                for j in cols:
                    if (x, x + w) in (cols) and (y, y + w) in (rows):
                        depth = depth_frame.get_distance(j, i)
                        # print(x, x+w, y, y+h, j, i)
                        depth_point = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [j, i], depth)
                        text = "%d: %.5lf, %.5lf, %.5lf\n" % (
                            curr_frame, depth_point[0], depth_point[1], depth_point[2])
                        f.write(text)
        print("Finish writing the depth img")

def run():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    curr_frame = 0

    cv2.getTickCount()

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            global ir_frame
            ir_frame = frames.get_infrared_frame()
            global depth_frame
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Intrinsics & Extrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
                color_frame.profile)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            isObstacleDetected, center_obstacle_point = obs_avoidance.detect_obstacle(color_image)
            if isObstacleDetected == 1:
                print("Obstacle detected. The mid position depth is:", depth_frame.get_distance(
                    int(center_obstacle_point[0]), int(center_obstacle_point[1])))
            else:
                print("Clear to fly")

            curr_frame += 1

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            #Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            #Exit on ESC key
            c = cv2.waitKey(1) % 0x100
            if c == 27:
                break

    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    run()