#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# RealSense Publisher Class                     #
# ********************************************* #

import sys
sys.path.insert(1, '/home/autonav/autonav_ws/src/autonav_utils')
sys.path.insert(1, '/home/autonav/librealsense/build/wrappers/python')

import cv2
import numpy as np
import rospy
import pyrealsense2 as rs
from autonav_node import AutoNavNode
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, MultiArrayLayout, MultiArrayDimension

# Run This Script: $ rosrun rs_publisher publish.py
# ROS Image Viewer: $ rosrun image_view image_view image:=/camera/color/image_raw

class RealsensePublisher(AutoNavNode):

    def __init__(self):
        AutoNavNode.__init__(self, "rs_publisher")

        # Read ROS Params
        self.read_param("/DisplayCameraImages", "DISPLAY", False)
        self.read_param("/ObjectCropBottom", "CROP_BOTTOM", 0.40)
        self.read_param("/ObjectCropTop", "CROP_TOP", 0.10)
        self.read_param("/ObjectCropSide", "CROP_SIDE", 0.30)
        self.read_param("/DanielDebug", "DANIEL_DEBUG", False)

        # Connect to RealSense Camera
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start()

        # Initialize Camera
        self.SCALE = 0.0010000000475
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        if abs(self.SCALE - depth_scale) >= 0.0000000000001:
            rospy.logerr("Depth scale changed to %.13f" % depth_scale)

        # Initialize Colorizer
        self.colorizer = rs.colorizer()

        # Publish camera data
        self.color_pub = rospy.Publisher("camera/color/image_raw", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("camera/depth/image_rect_raw", Float64MultiArray, queue_size=1)

    def remove_shadows(self, image):
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        (h, s, v) = cv2.split(img)

        cv2.imshow('h value', h)
        cv2.waitKey(2)
        cv2.imshow('s value', s)
        cv2.waitKey(2)
        cv2.imshow('v value', v)
        cv2.waitKey(2)

        #return result

    def shadow_remove(self, img):
        rgb_planes = cv2.split(img)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        shadowremov = cv2.merge(result_norm_planes)
        cv2.imshow('shadows_out', shadowremov)
        cv2.waitKey(2)
        return shadowremov

    def publish_color(self, frames):
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        #color_image = self.shadow_remove(color_image)
        #self.remove_shadows(color_image)

        if self.DANIEL_DEBUG:
            cv2.imshow("Original Color", cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV))
            cv2.waitKey(2)

        self.safe_publish(self.color_pub, self.bridge_image_pub(color_image, "rgb8"))
        # self.color_pub.publish(self.bridge_image_pub(color_image, "rgb8"))

    def publish_depth(self, frames):
        depth_frame = frames.get_depth_frame()

        if self.DISPLAY:
            depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            cv2.imshow("Original Depth", depth_image)
            cv2.waitKey(2)

        depth = np.asanyarray(depth_frame.get_data(), dtype=np.float64)
        depth = self.crop_depth(depth)

        # Create depth Float64MultiArray message to publish
        depth_data = Float64MultiArray()
        dim1 = MultiArrayDimension(label="height", size=depth.shape[0])
        dim2 = MultiArrayDimension(label="width", size=depth.shape[1])
        depth_data.layout.dim.extend([dim1, dim2])
        depth_data.data = depth.flatten()

        # Display Cropped Image
        if self.DANIEL_DEBUG and not self.DISPLAY:
            displayed_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
            cv2.imshow("Published Depth", self.crop_depth(displayed_depth))
            cv2.waitKey(2)

        self.safe_publish(self.depth_pub, depth_data)
        # self.depth_pub.publish(depth_data)

    def crop_depth(self, depth):
        y, x = depth.shape[0], depth.shape[1]
        return depth[int(y*self.CROP_TOP):-int(y*self.CROP_BOTTOM), int(x*self.CROP_SIDE):-int(x*self.CROP_SIDE)]

    def spin_detect(self):
        while not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            self.publish_color(frames)
            self.publish_depth(frames)


def main(args):
    rp = RealsensePublisher()

    try:
        rp.spin_detect()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
