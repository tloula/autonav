#!/usr/bin/env python

# ********************************************* #
# Cedarville University                         #
# AutoNav Senior Design Team 2020-2021          #
# Line Detection Class                          #
# ********************************************* #

import cv2
import numpy as np
from array import *

class LineFollowing:

    def __init__(self):

        # Class attributes
        self.MAX_DIST = 7777
        self.distance = self.MAX_DIST
        self.max_white = -1
        self.max_start = 800
        self.no_line_count = 0
        self.run = "go"
        self.LINE_CODE = "LIN,"


    # filters the resulting pixle distance with a 5 point moving filter
    def set_params(self, FollowingDirection, Debug, IsaiahDebug, LineBufferSize, LineThreshMin, LineThreshMax, LineHeightStart, LineHeightStep, LineLostCount):
        self.following_dir = FollowingDirection
        self.is_debut = Debug
        self.ISAIAH_DEBUG = IsaiahDebug
        self.BUFF_SIZE = LineBufferSize
        self.THRESH_MIN = LineThreshMin
        self.MAX_PIXEL = LineThreshMax
        self.HEIGHT_START = LineHeightStart
        self.HEIGHT_STEP = LineHeightStep
        self.LINE_LOST_COUNT = LineLostCount
        self.moving = array('i', [0] * self.BUFF_SIZE)
        self.display_window = IsaiahDebug


    def filter_result(self):
        self.moving = np.roll(self.moving, 1)
        self.moving[0] = self.distance
        filt_dist = sum(self.moving) / self.BUFF_SIZE
        return(filt_dist)

    def filter_image(self, image, x, y, w, h):
        
        # Convert image into thresholded greyscale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,grey =  cv2.threshold(grey, self.THRESH_MIN, self.MAX_PIXEL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Mask out portion of the image
        section = np.zeros(grey.shape, np.uint8)
        section[y:y+h, x:x+w] = grey[y:y+h, x:x+w]
        grey = section

        # Perform a morphological opening operation on the image
        # Uses a rectangular structuring element
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (int((7.0 / 1280.0) * grey.shape[0]), int((20 / 720.0) * grey.shape[1])))
        mask = cv2.morphologyEx(grey, cv2.MORPH_OPEN, element)
        return (mask)

    # returns the value in pixles that an assumed line is from the center of the image
    def follow_line(self, mask, begin, end, h, y):
            # Maximum values
            confidence = 100
            self.max_white = confidence
            
            # Width of the regin of interest
            # The divide and multiply allow this value to be used on any image size
            # The value is normalized in relation to a 1080 by 720 image, then put in reference to the current image dimmension
            w = int((self.HEIGHT_STEP / 720.0) * mask.shape[0])

            # Find the squre region with the highest white pixle count
            for x in range (begin, end, 1):
                white_count = cv2.countNonZero(mask[y:y+h, x:x+w])
                if white_count > self.max_white:
                    self.max_white = white_count
                    self.max_start = x

            # Obtain pixel distance
            # check to see if the max pixle count is substantial
            if self.max_white > confidence:
                new_dist = (self.max_start - (mask.shape[1]/2))
                # if there is a jump in the detectedc line position or if we lost the line previously, discard the reading
                if (abs(new_dist - self.distance) <= 5 * w) or (self.distance == self.MAX_DIST):
                    self.distance = new_dist
                    self.no_line_count = 0
            # we did not find a conclusive line in the image and will keep track of consecutive frames where there was no conclusion
            else:
                self.no_line_count += 1

            # report if we lost the line
            if not self.run or self.no_line_count == self.LINE_LOST_COUNT:
                self.run = False
                self.distance = self.MAX_DIST

            # 5 point moving average filter
            filt_dist =  self.filter_result()
            
            #return the resulting distance
            return(filt_dist)

    def printResult(self, mask, original_image, y, h, line_dist):
       
        # Creates and overlays a green square on the image where we think the line is
        # start pixles and width of square
        w = int((self.HEIGHT_STEP / 720.0) * mask.shape[0])
        self.max_start = line_dist + (mask.shape[1] / 2)

        # Create overlay
        pixels = np.array([[self.max_start, y], [self.max_start, y + h], [self.max_start + w, y + h], [self.max_start + w, y]], dtype=np.int32)
        overlay = cv2.fillConvexPoly(original_image, pixels, (43, self.MAX_PIXEL, 0))

        # Apply Overlay
        result = cv2.addWeighted(original_image, 1, overlay, 0.5, 0)

        # Display results
        cv2.imshow("result", result)
        cv2.waitKey(2)

    # This function takes an image, and returns the value of the distance (in pixles) that the line is 
    # from the center of the image
    def image_callback(self, original_image):

        # Reference variables to help find portion of image
        x = 0
        y = int((self.HEIGHT_START / 720.0) * original_image.shape[0])
        w = original_image.shape[1]
        h = int((self.HEIGHT_STEP / 720.0) * original_image.shape[0])

        # Mask out portion of the image
        # Set region of interest based on left or right follow
        if self.following_dir == 1:
            begin = int(0.5 * original_image.shape[1])
            end = original_image.shape[1]
        else:
            begin = 0
            end = int(0.5 * original_image.shape[1])

        # region of interest image to help find specific portion of the image
        section = np.zeros(original_image.shape,np.uint8)
        section[y:y+h, begin:end] = original_image[y:y+h, begin:end]


        # filter the line
        mask = self.filter_image(original_image, x, y, w, h)

        # find the line
        line_dist = self.follow_line(mask, begin, end, h, y)

        # print result for debug
        if self.display_window == True:
            self.printResult(mask, original_image, y, h, line_dist)
            #cv2.imshow("Line Location", section)
            #cv2.waitKey(2)
        
        return (abs(line_dist))