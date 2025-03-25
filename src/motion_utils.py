"""
Motion Detection Module for Face Tracking
Author: Romil V. Shah
This module provides robust motion detection between video frames with error recovery.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import config
import logging

logger = logging.getLogger(__name__)

def validate_frames(prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
    """
    Validate input frames for motion detection.
    
    Args:
        prev_frame: Previous frame numpy array
        curr_frame: Current frame numpy array
    
    Returns:
        bool: True if frames are valid for motion detection
    """
    if prev_frame is None or curr_frame is None:
        logger.error("Received None frame in motion detection")
        return False
    
    if prev_frame.shape != curr_frame.shape:
        logger.error(f"Frame size mismatch: {prev_frame.shape} vs {curr_frame.shape}")
        return False
    
    if len(prev_frame.shape) != 3 or prev_frame.dtype != np.uint8:
        logger.error("Invalid frame format, expected 3-channel BGR uint8")
        return False
    
    return True

def detect_motion(prev_frame: np.ndarray, 
                 curr_frame: np.ndarray,
                 debug: bool = False) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Detect motion between frames using optical flow with error handling.
    
    Args:
        prev_frame: Previous BGR frame
        curr_frame: Current BGR frame
        debug: Return visualization frame if True
    
    Returns:
        Optional[Tuple]: (magnitude array, visualization frame) or None
    """
    try:
        if not validate_frames(prev_frame, curr_frame):
            return None

        # Convert to grayscale with validation
        try:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            logger.error(f"Frame conversion failed: {str(e)}")
            return None

        # Calculate optical flow with error handling
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Calculate motion magnitude with clipping
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        vis_frame = None
        if debug or config.DEBUG_MODE:
            # Create HSV visualization for debug
            hsv = np.zeros_like(prev_frame)
            hsv[..., 1] = 255  # Max saturation
            
            # Convert angle from radians to degrees (0-180 for OpenCV)
            _, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = magnitude
            
            vis_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return (magnitude, vis_frame) if debug else magnitude

    except Exception as e:
        logger.error(f"Motion detection failed: {str(e)}")
        return None
