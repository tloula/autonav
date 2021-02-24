#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Line Detection Class                          #
# ********************************************* #

import sys
sys.path.insert(1, '/home/autonav/autonav_ws/src/autonav_utils')

import cv2
import numpy as np
import rospy

from performance_analyzer import PerformanceAnalyzerLite

class LineDetection:

    def __init__(self):
        self.history_idx = 0
        self.slope = None
        self.aligned = False
        self.found_line = False

        self.pa = PerformanceAnalyzerLite("general_purpose_pa")

    def set_params(self, is_debug, BUFF_SIZE, BUFF_FILL, CROP_TOP, CROP_BOTTOM, CROP_SIDE, MAX_WHITE, MIN_LINE_LENGTH, trevor_debug):
        self.is_debug = is_debug
        self.BUFF_SIZE = BUFF_SIZE
        self.BUFF_FILL = BUFF_FILL
        self.CROP_TOP = CROP_TOP
        self.CROP_BOTTOM = CROP_BOTTOM
        self.CROP_SIDE = CROP_SIDE
        self.MAX_WHITE = MAX_WHITE
        self.MIN_LINE_LENGTH = MIN_LINE_LENGTH
        self.line_history = [0] * self.BUFF_SIZE
        self.TREVOR_DEBUG = trevor_debug

    def image_callback(self, image):
        self.pa.start()
        # Save Dimensions
        y, x = image.shape[0], image.shape[1]

        # Slice Edges
        image = image[int(y*self.CROP_TOP):-int(y*self.CROP_BOTTOM), int(x*self.CROP_SIDE):-int(x*self.CROP_SIDE)]

        # Convert to Greyscale and Threshold
        ret, grey =  cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Static threshold
        #ret, grey = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 190, 255, cv2.THRESH_BINARY)

        # Discard Oversaturated Images
        if np.count_nonzero(grey) < image.shape[0]*image.shape[1]*self.MAX_WHITE:
            # Open Image
            morph = cv2.morphologyEx(grey, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            self.determine_line(morph)
            if self.TREVOR_DEBUG:
                cv2.imshow("Line Detection Opened Image", morph)
                cv2.waitKey(1)
        else:
            rospy.loginfo("Line Detection Image Discarded")
            self.update_history(0)

        # Display Image
        if self.TREVOR_DEBUG:
            #cv2.imshow("Line Detection Cropped Image", image)
            #cv2.waitKey(1)
            cv2.imshow("Line Detection Thresholded Image", grey)
            cv2.waitKey(1)

        found_line = self.determine_state()
        aligned = self.determine_orientation()

        self.pa.stop()
        return found_line, aligned

    def determine_line(self, image):
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 150, minLineLength=int(image.shape[0]*self.MIN_LINE_LENGTH), maxLineGap=40)
        if lines is not None:
            x1, y1, x2, y2 = lines[0][0]
            self.slope = float(abs(y2-y1)) / float(abs(x2-x1)+0.00001)
            self.update_history(1)
        else: self.update_history(0)

    def determine_state(self):
        if not self.found_line and self.line_history.count(1) >= self.BUFF_FILL * self.BUFF_SIZE:
            self.found_line = True
            return True
        return False

    def determine_orientation(self):
        if self.found_line and not self.aligned and self.slope >= 1:
            self.aligned = True
            return True
        return False

    def update_history(self, x):
        self.line_history[self.history_idx] = x
        self.history_idx = (self.history_idx + 1) % self.BUFF_SIZE

    def reset(self):
        self.line_history = [0] * self.BUFF_SIZE
        self.found_line = False
        self.aligned = False
