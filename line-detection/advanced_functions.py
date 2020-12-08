from datetime import datetime, timedelta
import numpy as np
import cupy as cp
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
from cv_bridge import CvBridge, CvBridgeError

##### CROP TOP OF IMAGE #####

def crop(image, y=0, x=0):
    h, w, _ = image.shape
    return image[y:h, x:w]

##### MASK ROBOT CHASSIS #####

def region_of_interest(image):
    height, width = image.shape

    # Polygon Points
    left_top            = (0, 0)
    left_bottom         = (0, height)
    left_lower_frame    = (int(0.27*width), height)
    left_upper_frame    = (int(0.29*width), int(0.65*height))
    right_upper_frame   = (int(0.75*width), int(0.65*height))
    right_lower_frame   = (int(0.82*width), height)
    right_bottom        = (width, height)
    right_top           = (width, 0)

    polygons = np.array(
        [[left_top, left_bottom, left_lower_frame, left_upper_frame,
        right_upper_frame, right_lower_frame, right_bottom, right_top]])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

##### THRESHOLD IMAGE #####

def threshold(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 190, 255, cv2.THRESH_BINARY)
    return thresh

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

##### SLIDING WINDOW ALGORITHM AND CURVE PLOTTING #####

right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=12, margin=100, minpix = 1, average_values=2, draw_windows=True):

    global right_a, right_b, right_c

    region_start_y = img.shape[0] - img.shape[0]/6
    region_end_y = img.shape[0] - img.shape[0]/30

    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    # Find peaks of the left and right halves of the histogram
    histogram = get_hist(img)
    midpoint = int(histogram.shape[0]/2)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()             # TODO WAY TOO SLOW!!!
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3)

        # Identify the nonzero pixels in x and y within the window
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    # Outputs constant coefficient values for second order polynomial y = ax**2 + bx + c
    right_fit, stats = np.polynomial.polynomial.polyfit(righty, rightx, 2, full=True)

    # Calculate R2 value
    R2 = stats[0][0]

    right_a.append(right_fit[2])
    right_b.append(right_fit[1])
    right_c.append(right_fit[0])

    # Calculate the current line position based on the average of it's past values
    right_fit_[0] = np.mean(right_a[-average_values:])
    right_fit_[1] = np.mean(right_b[-average_values:])
    right_fit_[2] = np.mean(right_c[-average_values:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    # Calculate distance from center
    line_region = right_fitx[region_start_y:region_end_y]
    line_x = sum(line_region)/len(line_region)

    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, right_fitx, right_fit_, ploty, histogram, line_x, R2

def draw_lanes(img, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)

    line_half_width = 0.015*img.shape[1]
    left_fit = [x - line_half_width for x in right_fit]
    right_fit = [x + line_half_width for x in right_fit]

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (149,0,179))
    overlay_img = cv2.addWeighted(img, 1, color_img, 1, 0)
    return overlay_img

##### VIDEO PIPELINE #####

# Save total callback times to display average
callback_times = []

def vid_pipeline(img_original):

    global callback_times
    output_times = False
    start_pipeline = datetime.now()

    # Crop Image
    start = datetime.now()
    img_crop = crop(img_original, 0, 0)
    if output_times: print("Cropping: {}".format(datetime.now() - start))

    # Threshold Image
    start = datetime.now()
    img_thresh = threshold(img_crop)
    if output_times: print("Thresholding: {}".format(datetime.now() - start))

    # Mask Region of Interest
    start = datetime.now()
    img_roi = region_of_interest(img_thresh)
    if output_times: print("Masking: {}".format(datetime.now() - start))

    # Histogram, Sliding Window Search, and Second Order Polynomial Fit
    start = datetime.now()
    out_img, curve, lanes, ploty, histogram, line_x, R2 = sliding_window(img_roi, draw_windows=True)
    if output_times: print("Sliding Window: {}".format(datetime.now() - start))

    # Overlay Line for Display
    start = datetime.now()
    img_overlay = draw_lanes(img_crop, curve)
    if output_times: print("Line Overlay: {}".format(datetime.now() - start))

    # Calculate distance using average of x pixels 600-700
    center = img_crop.shape[1]/2
    distance_from_line = line_x - center

    # Calculate R2 Confidence
    if (R2 < 15000000): r2_confidence = 1
    elif (R2 < 25000000): r2_confidence = .9
    elif (R2 < 40000000): r2_confidence = .25
    else: r2_confidence = 0

    # Calculate Histogram Confidence Using Constant Y Value of 60,000
    static_hist_confidence = min(max(histogram) * 0.0000166666667,  1)

    # Calculate Histogram Confidence Using Peak Relative to Edges
    padding = 25
    edge_magnitude = 6
    hist_max_index = np.where(histogram == max(histogram))[0][0]
    sum_max = sum(histogram[hist_max_index-padding:hist_max_index+padding])
    sum_edge = sum(histogram[hist_max_index-padding*edge_magnitude:hist_max_index-padding]) + sum(histogram[hist_max_index+padding:hist_max_index+padding*edge_magnitude])
    dynamic_hist_confidence = min(sum_max / sum_edge,  1)

    # Calculate Combined Confidence
    confidence = (r2_confidence + static_hist_confidence + dynamic_hist_confidence) / 3

    # Diagnostic Output
    #print("Stat Hist Confidence: {:.4f} | Dyn Hist Conf: {:.4f} | R2: {:.4f} | Avg: {:.4f}".format(
    #    static_hist_confidence, dynamic_hist_confidence, R2, confidence))

    # Overlay Distance on Display
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize=1
    text_pos_y = int(0.15*img_overlay.shape[0])
    text_pos_x = int(0.05*img_overlay.shape[1])
    cv2.putText(img_overlay, 'DISTANCE: {:.4f}px'.format(distance_from_line), (text_pos_x, text_pos_y), font, fontSize, fontColor, 2)
    cv2.putText(img_overlay, 'CONFIDENCE: {:.2f}%'.format(confidence*100), (text_pos_x, text_pos_y+50), font, fontSize, fontColor, 2)

    callback_times.append(datetime.now() - start_pipeline)
    if output_times: print("Total Pipeline: {}".format(sum(callback_times[-10:], timedelta(0))/len(callback_times[-10:])))

    #display(img_crop, img_roi, histogram, out_img, img_overlay, curve, ploty)

    return img_overlay, distance_from_line, confidence

##### Display #####

def display(img_original, img_filter, histogram, img_window, img_overlay, curve, ploty):

    plt.subplot(2, 3, 1)
    plt.imshow(img_original[...,::-1])  # RGB-> BGR
    plt.title('Original Image', fontsize=15)

    plt.subplot(2, 3, 2)
    plt.imshow(img_filter)
    plt.title('Thesholded Image', fontsize=15)

    plt.subplot(2, 3, 3)
    plt.plot(histogram)
    plt.title('Histogram of Y Columns', fontSize=15)

    plt.subplot(2, 3, 4)
    plt.imshow(img_window)
    plt.plot(curve, ploty, color='yellow', linewidth=5)
    plt.title('Sliding Windows + 2nd Order Curve Fit', fontsize=15)

    plt.subplot(2, 3, 5)
    plt.imshow(img_overlay[...,::-1])   # RGB-> BGR
    plt.title('Overlay Line', fontsize=15)

    plt.suptitle("Line Detection Process")

    plt.show()
    cv2.waitKey(0)
    plt.close()
