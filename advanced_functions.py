import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
from cv_bridge import CvBridge, CvBridgeError

##### MASK ROBOT CHASSIS #####

def region_of_interest(image):
    height, width = image.shape
    polygons = np.array(
        [[(0, 0), (0, height), (350, height), (400, 470),
        (950, 470), (1050, height), (width, height), (width, 0)]])
    mask = np.zeros_like(image)

    # Fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # Bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

##### PIPELINE AND PERSPECTIVE WARP #####

def pipeline(img, s_thresh=(175, 255), sx_thresh=(200, 255)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

##### SLIDING WINDOW ALGORITHM AND CURVE PLOTTING #####

right_a, right_b, right_c = [],[],[]

def sliding_window(img, nwindows=9, margin=100, minpix = 1, draw_windows=True):
    global right_a, right_b, right_c
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
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
    right_fit = np.polyfit(righty, rightx, 2)

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    # Calculate the current line position based on the average of the last 3 values
    right_fit_[0] = np.mean(right_a[-1:])
    right_fit_[1] = np.mean(right_b[-1:])
    right_fit_[2] = np.mean(right_c[-1:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, right_fitx, right_fit_, ploty

def draw_lanes(img, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)

    left_fit = [x - 20 for x in right_fit]
    right_fit = [x + 20 for x in right_fit]

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))

    cv2.fillPoly(color_img, np.int_(points), (43,255,0))
    overlay_img = cv2.addWeighted(img, 1, color_img, 0.7, 0)
    return overlay_img

##### VIDEO PIPELINE #####

def vid_pipeline(img_original):
    img_pipeline = pipeline(img_original)
    img_roi = region_of_interest(img_pipeline)
    out_img, curve, lanes, ploty = sliding_window(img_roi, draw_windows=True)

    # Calculate distance from line using the c value of the second order polynomial
    center = img_original.shape[1]/2
    distance_from_line = lanes[2] - center

    img_overlay = draw_lanes(img_original, curve)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (0, 0, 0)
    fontSize=1
    cv2.putText(img_overlay, 'CENTER TO LINE: {:.4f} px'.format(distance_from_line), (400, 650), font, fontSize, fontColor, 2)

    #display(img_original, img_roi, out_img, img_overlay, curve, ploty)

    return distance_from_line, img_overlay

##### Display #####

def display(img_original, img_filter, img_window, img_overlay, curve, ploty):

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(100, 20))

    ax1.imshow(img_original)
    ax1.set_title('Original', fontsize=15)

    ax2.imshow(img_filter)
    ax2.set_title('Filter + Perspective Transform', fontsize=15)

    ax3.imshow(img_window)
    ax3.plot(curve, ploty, color='yellow', linewidth=5)
    ax3.set_title('Sliding Window + Curve Fit', fontsize=15)

    ax4.imshow(img_overlay)
    ax4.set_title('Overlay Lanes', fontsize=15)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.0)
    plt.show()
