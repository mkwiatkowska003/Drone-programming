import pyrealsense2 as rs
import numpy as np
import cv2
import MobileNet as mobilenet


def get_only_depth():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    e1 = cv2.getTickCount()

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            return depth_colormap

    finally:

        # Stop streaming
        pipeline.stop()



def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    e1 = cv2.getTickCount()

    # Load the Caffe model
    net = cv2.dnn.readNetFromCaffe(
        "C:/Users/marta/PycharmProjects/realsense/MobileNet-SSD-RealSense-master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt",
        "C:/Users/marta/PycharmProjects/realsense/MobileNet-SSD-RealSense-master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel")

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

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
                    # Get the bounding boxes for the detections
                    box = mobilenet.draw_bounding_box(detections, color_image, key)

                    # Get predictions for the boxes
                    label = mobilenet.predict_class_labels(confidence, idx)

                    # Draw predictions on frames
                    mobilenet.draw_predictions_on_frames(box, label, color_image, idx)

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