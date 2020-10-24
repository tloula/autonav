#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Line Detection Class                          #
# ********************************************* #

# Import the required libraries
from cv_bridge import CvBridge, CvBridgeError
import cv2
from datetime import datetime
import rospy
from sensor_msgs.msg import Image, CameraInfo
import sys
from advanced_functions import vid_pipeline

# Instructions
# Start RealSense Publisher: $ roslaunch realsense2_camera rs_camera.launch
# Run This Script: $ rosrun lines line_detection_template.py

# ROS Image Viewer: $ rosrun image_view image_view image:=/camera/color/image_raw

class LineDetection:

    def __init__(self):

        self.node_name = "line_detection"
        rospy.init_node(self.node_name)

        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Save Lane Curvature Information
        right_curves, left_curves = [],[]

        # Subscribe to the camera image and depth topics and set the appropriate callbacks
        # self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, vid_pipeline, queue_size=1, buff_size=2**24)

        # Subscribe to prerecorded .bag files  
        self.image_sub = rospy.Subscriber("/device_0/sensor_1/Color_0/image/data/", Image, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):

        time_callback = datetime.now()

        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            original_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge could not convert images from realsense to opencv")

        # Original Image
        #cv2.imshow("Original Image", original_image)
        #cv2.waitKey(2)

        # Run Advanced Line Detection
        distance, processed_image = vid_pipeline(original_image)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(2)

        #print ("total callback", datetime.now() - time_callback)

    def cleanup(self):
        print "Shutting down vision node."
        cv2.destroyAllWindows()  

def main(args):
    ld = LineDetection()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
