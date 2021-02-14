#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Line Controller Class                         #
# ********************************************* #

import sys
sys.path.insert(1, '/home/autonav/autonav_ws/src/autonav_utils')

import rospy
from line_detection import LineDetection
from line_following import LineFollowing
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from autonav_node import AutoNavNode

class Lines(AutoNavNode):

    def __init__(self):
        AutoNavNode.__init__(self, "lines")

        # States
        self.FOUND_LINE = "FOUND_LINE"
        self.LOST_LINE = "LOST_LINE"
        self.ALIGNED = "ALIGNED"
        self.NOT_ALIGNED = "NOT_ALIGNED"

        # Read ROS Params - Line Detection
        self.read_param("/LineDetectBufferSize", "DETECT_BUFF_SIZE", 10)
        self.read_param("/LineDetectBufferFill", "DETECT_BUFF_FILL", 0.8)
        self.read_param("/LineDetectCropBottom", "CROP_BOTTOM", 0.52)
        self.read_param("/LineDetectCropSide", "CROP_SIDE", 0.20)

        # Read ROS Params - Line Following
        self.read_param("/IsaiahDebug", "ISAIAH_DEBUG", False)
        self.read_param("/LineBufferSize", "FOLLOW_BUFF_SIZE", 5)
        self.read_param("/LineThreshMin", "THRESH_MIN", 250)
        self.read_param("/LineThreshMax", "MAX_PIXEL", 255)
        self.read_param("/LineHeightStart", "HEIGHT_START", 470.0)
        self.read_param("/LineHeightStep", "HEIGHT_STEP", 50.0)
        self.read_param("/LineLostCount", "LINE_LOST_COUNT", 100)

        self.LINE_CODE = "LIN,"

        # Subscribe to the camera color image
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        # Publish events that could change the robot state
        self.event_pub = rospy.Publisher("line_events", String, queue_size=10)

        # Publish distance from the center of the line
        self.motor_pub = rospy.Publisher("wheel_distance", String, queue_size=10)

        # Initialize Classes
        self.line_detection = LineDetection()
        self.line_following = LineFollowing()

        # Set ROS Params
        self.line_detection.set_params(
            self.is_debug,
            self.DETECT_BUFF_SIZE,
            self.CROP_BOTTOM,
            self.CROP_SIDE)
        self.line_following.set_params(
            self.following_dir,
            self.is_debug,
            self.ISAIAH_DEBUG,
            self.FOLLOW_BUFF_SIZE,
            self.THRESH_MIN,
            self.MAX_PIXEL,
            self.HEIGHT_START,
            self.HEIGHT_STEP,
            self.LINE_LOST_COUNT)

        rospy.loginfo("Waiting for image topics...")

    def image_callback(self, ros_image):
        image = self.bridge_image(ros_image, "bgr8")

        # Line Following    
        if self.state == self.CHANGE_TO_LINE_FOLLOWING:
            distance = self.line_following.image_callback(image)
            rospy.loginfo("LINE_DISTANCE: " + str(distance))
            self.motor_pub.publish(self.LINE_CODE + str(distance))

        # Line Detection
        elif self.state == self.CHANGE_TO_OBJECT_AVOIDENCE_FROM_LINE:
            found_updated, found_line, aligned_updated, aligned = self.line_detection.image_callback(image)
            if found_updated and found_line:
                rospy.loginfo(self.FOUND_LINE)
                self.event_pub.publish(self.FOUND_LINE)
            elif found_updated:
                rospy.loginfo(self.LOST_LINE)
                self.event_pub.publish(self.LOST_LINE)

            if aligned_updated and aligned:
                rospy.loginfo(self.ALIGNED)
                self.event_pub.publish(self.ALIGNED)
            elif aligned_updated:
                rospy.loginfo(self.NOT_ALIGNED)
                self.event_pub.publish(self.NOT_ALIGNED)

def main(args):
    lines = Lines()

    try: rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt")
    finally:
        rospy.loginfo("Shutting down line class")

if __name__ == "__main__":
    main(sys.argv)