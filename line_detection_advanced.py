#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Line Detection Class                          #
# ********************************************* #

# Import the required libraries
from advanced_functions import vid_pipeline
from cv_bridge import CvBridge, CvBridgeError
import cv2
from datetime import datetime
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
import sys

# Start RealSense Publisher: $ roslaunch realsense2_camera rs_camera.launch
# Run This Script: $ rosrun lines line_detection_advanced.py

# ROS Image Viewer: $ rosrun image_view image_view image:=/camera/color/image_raw

class LineDetection:

    def __init__(self):

        self.node_name = "line_detection"
        rospy.init_node(self.node_name)

        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Subscribe to the camera image topics and set the appropriate callbacks
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        # Publish distance from line to wheel_distance topic to be read by motor controllers
        self.motor_pub = rospy.Publisher("wheel_distance", Float32, queue_size=10) # consider changing queue_size

        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):

        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            original_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge could not convert images from realsense to opencv")

        # Run Advanced Line Detection
        img_overlay, distance, confidence = vid_pipeline(original_image)
        cv2.imshow("Processed Image", img_overlay)
        cv2.waitKey(2)

        # Send Stop Command If Lost Line
        if (confidence < .55): distance = 7777

        # Publish distance to motor controller
        self.motor_pub.publish(distance)
        rospy.loginfo(distance)
        with open("/home/autonav/Documents/better_distances.csv", 'a+') as f:
            f.write("%s," % distance)

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
