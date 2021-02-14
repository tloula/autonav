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

class LineDetection:

    def __init__(self):
        self.history_idx = 0
        self.slope = None
        self.aligned = False
        self.found_line = False

    def set_params(self, is_debug, BUFF_SIZE, BUFF_FILL, CROP_BOTTOM, CROP_SIDE):
        self.is_debug = is_debug
        self.BUFF_SIZE = BUFF_SIZE
        self.BUFF_FILL = BUFF_FILL
        self.CROP_BOTTOM = CROP_BOTTOM
        self.CROP_SIDE = CROP_SIDE
        self.line_history = [0] * self.BUFF_SIZE

    def image_callback(self, image):
        # Save Dimensions
        y, x = image.shape[0], image.shape[1]

        # Slice Edges
        image = image[:y-int(y*self.CROP_BOTTOM), int(x*self.CROP_SIDE):x-int(x*self.CROP_SIDE)]

        # Convert to Greyscale and Threshold
        ret, grey =  cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Open Image
        morph = cv2.morphologyEx(grey, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        # Display Image
        if self.is_debug:
            cv2.imshow("Thresholded Image", grey)
            cv2.waitKey(1)
            cv2.imshow("Opened Image", morph)
            cv2.waitKey(1)

        self.determine_line(morph)
        found_updated = self.determine_state()
        aligned_updated = self.determine_orientation()

        return found_updated, self.found_line, aligned_updated, self.aligned

    def determine_line(self, image):
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 150, minLineLength=250, maxLineGap=40)
        if lines is not None:
            x1, y1, x2, y2 = lines[0][0]
            self.slope = float(abs(y2-y1)) / float(abs(x2-x1)+0.00001)
            self.line_history[self.history_idx] = 1
        else:
            self.line_history[self.history_idx] = 0
        self.history_idx = (self.history_idx + 1) % self.BUFF_SIZE

    def determine_state(self):
        if not self.found_line and self.line_history.count(1) >= self.BUFF_FILL * self.BUFF_SIZE:
            self.found_line = True
            return True
        elif self.found_line and self.line_history.count(0) >= self.BUFF_FILL * self.BUFF_SIZE:
            self.found_line = False
            return True
        return False

    def determine_orientation(self):
        if self.found_line and not self.aligned and self.slope >= 1:
            self.aligned = True
            return True
        elif self.found_line and self.aligned and self.slope < 1:
            self.aligned = False
            return True
        return False
