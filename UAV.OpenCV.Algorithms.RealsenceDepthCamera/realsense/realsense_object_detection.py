import pyrealsense2 as rs
import numpy as np
import cv2
import os
import MobileNet as mobilenet


def get_mobilenet_model():
    # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(
        "C:/Users/marta/Desktop/repos/Drone-programming/UAV.OpenCV.Algorithms.RealsenceDepthCamera/realsense/MobileNet-SSD-RealSense-master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt",
        "C:/Users/marta/Desktop/repos/Drone-programming/UAV.OpenCV.Algorithms.RealsenceDepthCamera/realsense/MobileNet-SSD-RealSense-master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel")

    return net


def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    curr_frame = 0

    cv2.getTickCount()
    net = get_mobilenet_model()

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
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

            blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            detections = mobilenet.get_detections_from_frames(blob, net)

            # loop over the detections
            for key in np.arange(0, detections.shape[2]):
                # extract the confidence associated with the prediction
                confidence = detections[0, 0, key, 2]

                # filter out weak detections
                args = mobilenet.agr_parser()
                idx = mobilenet.filter_out_detections(args, detections, confidence, key)

                if (idx == None):
                    break
                else:
                    if confidence > 0.5:
                        # Get the bounding boxes for the detections
                        box = mobilenet.draw_bounding_box(detections, color_image, key)

                        # Get predictions for the boxes
                        label = mobilenet.predict_class_labels(confidence, idx)

                        # Draw predictions on frames
                        mobilenet.draw_predictions_on_frames(box, label, color_image, idx)

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
                                        if (x, x+w) in (cols) and (y, y+w) in (rows):
                                            depth = depth_frame.get_distance(j, i)
                                            #print(x, x+w, y, y+h, j, i)
                                            depth_point = rs.rs2_deproject_pixel_to_point(
                                                depth_intrin, [j, i], depth)
                                            text = "%d: %.5lf, %.5lf, %.5lf\n" % (
                                                curr_frame, depth_point[0], depth_point[1], depth_point[2])
                                            f.write(text)
                            print("Finish writing the depth img")

                        #cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            curr_frame += 1

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # Exit on ESC key
            c = cv2.waitKey(1) % 0x100
            if c == 27:
                break

    finally:

        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()