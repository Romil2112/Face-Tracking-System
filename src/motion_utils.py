"""
Motion Detection Module for Face Tracking
Author: Romil V. Shah
This module provides utilities for detecting motion between video frames.
"""

import cv2
import numpy as np

def detect_motion(prev_frame, curr_frame):
    """
    Detect motion between frames to verify if a face is actually moving.
    """
    if prev_frame is None or curr_frame is None:
        return None
        
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return magnitude
