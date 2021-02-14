#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Obstacle Detection Class                      #
# ********************************************* #

import sys
sys.path.insert(1, '/home/autonav/autonav_ws/src/autonav_utils')

import cv2
import numpy as np
import rospy
from autonav_node import AutoNavNode
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float64MultiArray

# Start RealSense Publisher: $ roslaunch realsense2_camera rs_camera.launch aligned_depth:=true
# Run This Script: $ rosrun depth ob_detection_basic.py
# ROS Image Viewer: $ rosrun image_view image_view image:=/camera/depth/image_rect_raw

class ObstacleDetection(AutoNavNode):

    def __init__(self):
        AutoNavNode.__init__(self, "ob_detection")

        # States
        self.OBJECT_SEEN = "OBJECT_SEEN"
        self.PATH_CLEAR = "PATH_CLEAR"

        # Read ROS params -- read_param func params are ROS_Param, var_name, default_val
        self.read_param("/ObjectBufferSize", "BUFF_SIZE", 3)
        self.read_param("/ObjectBufferFill", "BUFF_FILL", 0.8)
        self.read_param("/ObjectStopDistance", "DISTANCE_AT_WHICH_WE_STOP_FROM_THE_OBSTACLE", 1.5)

        # Publish events that could change the robot state
        self.event_pub = rospy.Publisher("depth_events", String, queue_size=10)

        # Subscribe to the camera depth topic
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Float64MultiArray, self.image_callback, queue_size=1, buff_size=2**24)

        # Scale to convert depth image to distance
        self.DEPTH_SCALE = 0.0010000000475

        # Initialize primary variables
        self.object_history = [0] * self.BUFF_SIZE
        self.history_idx = 0
        self.path_clear = True

    def get_hist(self, img):
        hist = np.sum(img, axis=0)
        return hist

    def determine_obstacle(self, min_distance):
        if min_distance < self.DISTANCE_AT_WHICH_WE_STOP_FROM_THE_OBSTACLE:
            self.object_history[self.history_idx] = 1
        else:
            self.object_history[self.history_idx] = 0
        self.history_idx = (self.history_idx + 1) % self.BUFF_SIZE

    def determine_state(self):
        if self.path_clear and self.object_history.count(1) >= self.BUFF_FILL * self.BUFF_SIZE:
            rospy.loginfo(self.OBJECT_SEEN)
            self.event_pub.publish(self.OBJECT_SEEN)
            self.path_clear = False
        elif not self.path_clear and self.object_history.count(0) >= self.BUFF_FILL * self.BUFF_SIZE:
            rospy.loginfo(self.PATH_CLEAR)
            self.event_pub.publish(self.PATH_CLEAR)
            self.path_clear = True

    def image_callback(self, depth_arr):
        # Reshape 1-D list back to 2-D numpy array
        height, width = depth_arr.layout.dim[0].size, depth_arr.layout.dim[1].size
        depth_arr = np.asanyarray(depth_arr.data, dtype=np.float64).reshape(height, width)

        # Determine Pixel Distances
        distances = depth_arr * self.DEPTH_SCALE
        distances[distances == 0] = 10 #TODO check if distance < value or just 0?

        # Display Post-Processed Image TODO: modify to be displayed better (could colorize and display here or in publisher.py)
        # cv2.imshow("Post Image", depth_arr)
        # cv2.waitKey(2)

        min_dist = np.min(distances)
        if self.is_debug: print("Distance: {:.2f} | Object: {}".format(min_dist, not self.path_clear))

        self.determine_obstacle(min_dist)
        self.determine_state()


def main(args):
    od = ObstacleDetection()

    try: rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
