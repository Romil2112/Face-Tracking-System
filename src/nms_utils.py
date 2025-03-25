"""
Non-Maximum Suppression Module for Face Tracking
Author: Romil V. Shah
This module provides utilities for applying Non-Maximum Suppression to filter overlapping face detections.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict
import config

logger = logging.getLogger(__name__)

def apply_nms(face_info: List[Dict], overlap_threshold: float = config.NMS_THRESHOLD) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping face detections.

    Args:
        face_info: List of dictionaries containing face information
                   with 'rect' (x,y,w,h) and 'confidence' keys
        overlap_threshold: Minimum overlap ratio to consider as duplicate

    Returns:
        Filtered list of face information dictionaries
    """
    # Validate input
    if not face_info or any('rect' not in face for face in face_info):
        return []

    # Extract rectangles and confidence scores
    boxes = []
    scores = []
    valid_faces = []
    
    for face in face_info:
        try:
            x, y, w, h = face['rect']
            boxes.append([x, y, x + w, y + h])
            scores.append(face.get('confidence', 0.5))
            valid_faces.append(face)
        except Exception as e:
            logger.warning(f"Invalid face entry: {str(e)}")

    if len(valid_faces) <= 1:
        return valid_faces

    try:
        # OpenCV 4.x+ NMSBoxes API
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            scores,
            score_threshold=config.MINIMUM_CONFIDENCE,
            nms_threshold=overlap_threshold
        )
        
        # Handle different return types across OpenCV versions
        if indices is not None:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            return [valid_faces[i] for i in indices]
        return []

    except (cv2.error, ValueError) as e:
        logger.error(f"NMS failed: {str(e)}. Using manual IOU filtering.")
        return _manual_iou_filter(valid_faces, overlap_threshold)

def _manual_iou_filter(faces: List[Dict], threshold: float) -> List[Dict]:
    """Fallback IOU-based filtering"""
    filtered = []
    for current in sorted(faces, key=lambda x: x['confidence'], reverse=True):
        keep = True
        for kept in filtered:
            iou = _calculate_iou(current['rect'], kept['rect'])
            if iou > threshold:
                keep = False
                break
        if keep:
            filtered.append(current)
    return filtered

def _calculate_iou(box1, box2):
    """Calculate Intersection over Union (reused from temporal_filter.py)"""
    # Convert to (x1, y1, x2, y2)
    box1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
    box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]
    
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection)
