"""
Non-Maximum Suppression Module for Face Tracking
Author: Romil V. Shah
This module provides utilities for applying Non-Maximum Suppression to filter overlapping face detections.
"""


import numpy as np
import cv2
from typing import List, Tuple, Dict

def apply_nms(face_info: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping face detections.
    
    Args:
        face_info: List of dictionaries containing face information 
                  with 'rect' and 'area' keys
        overlap_threshold: Minimum overlap ratio to consider as duplicate
        
    Returns:
        Filtered list of face information dictionaries
    """
    # If no faces or only one face, return as is
    if len(face_info) <= 1:
        return face_info
        
    # Extract rectangles and scores
    boxes = np.array([list(face['rect']) for face in face_info])
    scores = np.array([face['area'] for face in face_info])
    
    # Convert to format expected by NMSBoxes: [x, y, x+w, y+h]
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    boxes_for_nms = np.column_stack((x, y, x + w, y + h))
    
    # Apply NMS
    try:
        # OpenCV 4.x API
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(), 
            scores.tolist(), 
            score_threshold=0.0, 
            nms_threshold=overlap_threshold
        )
        
        # OpenCV 4.x returns a 2D array, need to flatten
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        # Newer versions may return a 1D array directly
        elif isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], np.ndarray):
            indices = [item[0] for item in indices]
            
    except Exception as e:
        print(f"NMS error: {e}. Falling back to manual filtering.")
        return face_info
    
    # Keep only the selected indices
    return [face_info[i] for i in indices]
