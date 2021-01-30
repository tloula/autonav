#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Obstacle Detection Class                      #
# ********************************************* #

import cv2
from cv_bridge import CvBridge, CvBridgeError
#from datetime import datetime
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
#from std_msgs.msg import Float32
from std_msgs.msg import String
import sys

OBJECT_SEEN = "OBJECT_SEEN"
PATH_CLEAR = "PATH_CLEAR"
CHANGE_TO_LINE_FOLLOWING = "LINE_FOLLOWING_STATE"
CHANGE_TO_OBJECT_AVOIDENCE_FROM_LINE = "OBJECT_AVOIDENCE_FROM_LINE_STATE"
CHANGE_TO_OBJECT_AVOIDENCE_FROM_GPS = "OBJECT_AVOIDENCE_FROM_GPS_STATE"
CHANGE_TO_GPS_NAVIGATION = "GPS_NAVIGATION_STATE"
CHANGE_TO_LINE_TO_OBJECT = "LINE_TO_OBJECT_STATE"

# Start RealSense Publisher: $ roslaunch realsense2_camera rs_camera.launch aligned_depth:=true
# Run This Script: $ rosrun depth ob_detection_basic.py
# ROS Image Viewer: $ rosrun image_view image_view image:=/camera/depth/image_rect_raw

class ObstacleDetection:

    def __init__(self):

        # Initialize node
        self.node_name = "ob_detection"
        rospy.init_node(self.node_name)
        self.read_params()

        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Subscribe to state updates for the robot
        self.state_sub = rospy.Subscriber("state_topic", String, self.state_callback)

        # Publish events that could change the robot state
        self.event_pub = rospy.Publisher("depth_events", String, queue_size=10)

        # Subscribe to the camera depth topic
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        # Initialze object detection publisher - DO WE NEED THIS HERE?
        # self.motor_pub = rospy.Publisher("ob_distance", Float32, queue_size=10) # consider changing queue_size

        # Initialize primary variables
        self.object_history = [0] * self.BUFF_SIZE
        self.history_idx = 0
        self.path_clear = True

        rospy.loginfo("Waiting for image topics...")

    def state_callback(self, new_state):
        rospy.loginfo("New State Received (depth node)")
        pass

    def image_callback(self, ros_image):
        # Convert Image
        cv_depth_image = self.bridge_image(ros_image)

        # Normalize Image
        #cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
        #cv_depth_image = cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)

        # Save Dimensions
        x, y = cv_depth_image.shape[1], cv_depth_image.shape[0]

        # Slice Edges
        cv_depth_image = cv_depth_image[int(y*self.CROP_TOP):y-int(y*self.CROP_BOTTOM),
            int(x*self.CROP_SIDE):x-int(x*self.CROP_SIDE)]

        # Threshold Image
        ostu_thresh_value, cv_depth_image = cv2.threshold(cv_depth_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Open Image
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
        cv_depth_image = cv2.morphologyEx(cv_depth_image, cv2.MORPH_OPEN, element)

        # Display Image
        cv2.imshow("original", cv_depth_image)
        cv2.waitKey(2)

        self.determine_obstacle(cv_depth_image)
        self.determine_state()

    # Use cv_bridge() to convert the ROS image to OpenCV format
    def bridge_image(self, ros_image):
        # Switched to "8UC1" from "passthrough"
        try: cv_image = self.bridge.imgmsg_to_cv2(ros_image, "8UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge could not convert images from realsense to opencv")
        return cv_image

    def determine_obstacle(self, image):
        object_count = cv2.countNonZero(image[:image.shape[0]-int(self.CHASSIS_HEIGHT*image.shape[0]), :])
        if object_count < self.WHITE_MAX:
            if object_count > self.WHITE_MIN:
                #print("Object Detected | {} Pixels".format(object_count))
                self.object_history[self.history_idx] = 1
            else:
                #print("None | {} Pixels".format(object_count))
                self.object_history[self.history_idx] = 0
            self.history_idx = (self.history_idx + 1) % self.BUFF_SIZE

    def determine_state(self):
        if self.path_clear and self.object_history.count(1) >= 0.80 * self.BUFF_SIZE:
                rospy.loginfo(OBJECT_SEEN)
                self.event_pub.publish(OBJECT_SEEN)
                self.path_clear = False
        elif not self.path_clear and self.object_history.count(0) >= 0.80 * self.BUFF_SIZE:
                rospy.loginfo(PATH_CLEAR)
                self.event_pub.publish(PATH_CLEAR)
                self.path_clear = True

    # Read param from params.yaml and set var_name to value from file or default_val
    def read_param(self, param, var_name, default_val):
        if rospy.has_param(param):
            setattr(self, var_name, rospy.get_param(param))
        else:
            rospy.logwarn("%s not found. Using default instead. (%s)" % (param, default_val))
            setattr(self, var_name, default_val)

    # Function for reading ROS params
    def read_params(self):
        self.read_param("/FollowingDirection", "following_dir", 1)
        self.read_param("/Debug", "is_debug", False)
        self.read_param("/ObjectBufferSize", "BUFF_SIZE", 5)
        self.read_param("/ObjectCropBottom", "CROP_BOTTOM", 0.250)
        self.read_param("/ObjectCropTop", "CROP_TOP", 0.10417)
        self.read_param("/ObjectCropSide", "CROP_SIDE", 0.353774)
        self.read_param("/ObjectChassisHeight", "CHASSIS_HEIGHT", 0.25)
        self.read_param("/ObjectWhiteMin", "WHITE_MIN", 1000)
        self.read_param("/ObjectWhiteMax", "WHITE_MAX", 60000)

    def cleanup(self):
        print("Shutting down depth node")
        cv2.destroyAllWindows()


def main(args):
    od = ObstacleDetection()

    try: rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
