#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Top-Level ROS Node Class                      #
# ********************************************* #

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

class AutoNavNode:

    def __init__(self, name):
        # States
        self.CHANGE_TO_LINE_FOLLOWING = "LINE_FOLLOWING_STATE"
        self.CHANGE_TO_OBJECT_AVOIDENCE_FROM_LINE = "OBJECT_AVOIDENCE_FROM_LINE_STATE"
        self.CHANGE_TO_OBJECT_AVOIDENCE_FROM_GPS = "OBJECT_AVOIDENCE_FROM_GPS_STATE"
        self.CHANGE_TO_GPS_NAVIGATION = "GPS_NAVIGATION_STATE"
        self.CHANGE_TO_LINE_TO_OBJECT = "LINE_TO_OBJECT_STATE"
        self.state = self.CHANGE_TO_LINE_FOLLOWING

        # Read ROS Params
        self.read_param("/FollowingDirection", "following_dir", 1)
        self.read_param("/Debug", "is_debug", False)

        self.node_name = name
        rospy.init_node(self.node_name)
        
        # What we do during shutdown
        rospy.on_shutdown(self.cleanup)

        # Create the cv_bridge object
        self.bridge = CvBridge()

        # Subscribe to state updates for the robot
        self.state_sub = rospy.Subscriber("state_topic", String, self.state_callback)

    def state_callback(self, new_state):
        rospy.loginfo("New State Received (" + self.node_name + ")")
        self.state = new_state

    # Use cv_bridge() to convert the ROS image to OpenCV format
    def bridge_image(self, ros_image, format):
        try: cv_image = self.bridge.imgmsg_to_cv2(ros_image, format)
        except CvBridgeError as e: rospy.logerr("CvBridge could not convert images from ROS to OpenCV")
        return cv_image

    def bridge_image_pub(self, cv_image, format):
        try: ros_image = self.bridge.cv2_to_imgmsg(cv_image, format)
        except CvBridgeError as e: rospy.logerr("CvBridge could not convert images from OpenCV to ROS")
        return ros_image

    # Read param from params.yaml and set var_name to value from file or default_val
    def read_param(self, param, var_name, default_val):
        if rospy.has_param(param):
            setattr(self, var_name, rospy.get_param(param))
        else:
            rospy.logwarn("%s not found. Using default instead. (%s)" % (param, default_val))
            setattr(self, var_name, default_val)

    def cleanup(self):
        print("Shutting down " + self.node_name + " node")
        cv2.destroyAllWindows()
