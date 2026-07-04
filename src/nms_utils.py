"""
Non-Maximum Suppression Module for Face Tracking
Author: Romil V. Shah
This module provides utilities for applying Non-Maximum Suppression to filter overlapping face detections.
"""

import logging

import cv2
import numpy as np

import config
from geometry import calculate_iou

logger = logging.getLogger(__name__)

def apply_nms(face_info: list[dict], overlap_threshold: float = config.NMS_THRESHOLD) -> list[dict]:
    """
    Apply Non-Maximum Suppression to filter overlapping face detections.

    Args:
        face_info: List of dictionaries containing face information
                   with 'rect' (x,y,w,h) and 'confidence' keys
        overlap_threshold: Minimum overlap ratio to consider as duplicate

    Returns:
        Filtered list of face information dictionaries
    """
    boxes, scores, valid_faces = _prepare_boxes(face_info)
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
    except (cv2.error, ValueError) as e:
        logger.error(f"NMS failed: {str(e)}. Using manual IOU filtering.")
        return _manual_iou_filter(valid_faces, overlap_threshold)

    # Handle different return types across OpenCV versions.
    if indices is None:
        return []
    indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
    return [valid_faces[i] for i in indices]


def _prepare_boxes(face_info: list[dict]):
    """Extract (boxes, scores, valid_faces) from raw detections.

    Rejects the whole batch if any entry lacks 'rect'; skips individual entries
    whose rect can't be unpacked. Boxes are (x1, y1, x2, y2) for cv2.dnn.NMSBoxes.
    """
    if not face_info or any('rect' not in face for face in face_info):
        return [], [], []
    boxes, scores, valid_faces = [], [], []
    for face in face_info:
        try:
            x, y, w, h = face['rect']
            boxes.append([x, y, x + w, y + h])
            scores.append(face.get('confidence', 0.5))
            valid_faces.append(face)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Invalid face entry: {str(e)}")
    return boxes, scores, valid_faces

def _manual_iou_filter(faces: list[dict], threshold: float) -> list[dict]:
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
    """IoU of two (x, y, w, h) boxes. Thin wrapper over geometry.calculate_iou."""
    return calculate_iou(box1, box2)
