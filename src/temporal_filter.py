"""
Temporal Filtering Module for Face Tracking
Author: Romil V. Shah
This module implements temporal filtering to reduce false positives in face tracking.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)

class TemporalFilter:
    """
    Class for filtering face detections across multiple frames with error recovery.
    """

    def __init__(self, history_size: int = 5, consistency_threshold: int = 3):
        """
        Initialize the temporal filter with validation.
        Args:
            history_size (int): Number of frames to track history (1-50)
            consistency_threshold (int): Minimum consistent detections required (1-10)
        """
        if not 1 <= history_size <= 50:
            raise ValueError("History size must be between 1 and 50")
        if not 1 <= consistency_threshold <= 10:
            raise ValueError("Consistency threshold must be between 1 and 10")
        
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold
        self.face_history = deque(maxlen=history_size)

    def _validate_face(self, face: Dict) -> bool:
        """Validate face dictionary structure."""
        required_keys = {'rect', 'center', 'confidence'}
        if not all(key in face for key in required_keys):
            logger.warning("Invalid face structure detected")
            return False
        x, y, w, h = face['rect']
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid face dimensions: {face['rect']}")
            return False
        return True

    def _calculate_iou(self, box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union with validation.
        Args:
            box1, box2: Bounding boxes in (x, y, w, h) format
        Returns:
            float: IoU score between 0 and 1
        """
        try:
            # Convert to (x1, y1, x2, y2) format
            box1 = (box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3])
            box2 = (box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3])
            
            # Calculate intersection coordinates
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            logger.error(f"IoU calculation error: {str(e)}")
            return 0.0

    def _match_faces(self, current_faces: List[Dict],
                     previous_faces: List[Dict]) -> List[int]:
        """
        Match faces between frames using IoU and confidence.
        Args:
            current_faces: Current frame detections
            previous_faces: Previous frame detections
        Returns:
            List of matched indices in previous frame (-1 for no match)
        """
        matches = []
        for current_face in current_faces:
            if not self._validate_face(current_face):
                matches.append(-1)
                continue
            
            best_match = -1
            best_score = -1.0
            for idx, prev_face in enumerate(previous_faces):
                if not self._validate_face(prev_face):
                    continue
                
                iou = self._calculate_iou(current_face['rect'], prev_face['rect'])
                confidence_score = (current_face['confidence'] + prev_face['confidence']) / 2
                combined_score = iou * confidence_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = idx
            
            matches.append(best_match if best_score > 0.3 else -1)
        return matches

    def _calculate_motion(self, faces: List[Dict]) -> np.ndarray:
        """Calculate motion vector based on face positions."""
        if len(self.face_history) < 2:
            return np.zeros((len(faces), 2))
        
        prev_faces = self.face_history[-2]
        matches = self._match_faces(faces, prev_faces)
        motion_vector = np.zeros((len(faces), 2))
        
        for i, match in enumerate(matches):
            if match != -1:
                current_center = np.array(faces[i]['center'])
                prev_center = np.array(prev_faces[match]['center'])
                motion_vector[i] = current_center - prev_center
        
        return motion_vector

    def _apply_motion_prediction(self, faces: List[Dict], motion_vector: np.ndarray) -> List[Dict]:
        """Apply motion prediction to face positions."""
        for i, face in enumerate(faces):
            predicted_center = np.array(face['center']) + motion_vector[i]
            face['predicted_center'] = tuple(map(int, predicted_center))
        return faces

    def update(self, faces: List[Dict]) -> List[Dict]:
        """
        Update face history and apply temporal filtering with error handling.
        Args:
            faces (List[Dict]): Current frame face detections
        Returns:
            List[Dict]: Filtered face detections
        """
        try:
            if not isinstance(faces, list):
                logger.error("Invalid input type for faces")
                return []
            
            # Validate and filter invalid faces
            valid_faces = [face for face in faces if self._validate_face(face)]
            
            # Update face history
            self.face_history.append(valid_faces)
            
            # Apply motion prediction if enough history
            if len(self.face_history) >= 2:
                motion_vector = self._calculate_motion(valid_faces)
                valid_faces = self._apply_motion_prediction(valid_faces, motion_vector)
            
            # Maintain consistency tracking
            for face in valid_faces:
                if not self.face_history:
                    face['consistency'] = 1
                    continue
                
                matches = self._match_faces([face], self.face_history[-1])
                if matches and matches[0] != -1:
                    face['consistency'] = self.face_history[-1][matches[0]].get('consistency', 1) + 1
                else:
                    face['consistency'] = 1
            
            # Filter based on consistency threshold
            filtered_faces = [
                face for face in valid_faces
                if face.get('consistency', 0) >= self.consistency_threshold
            ]
            
            return filtered_faces
        except Exception as e:
            logger.error(f"Temporal filtering error: {str(e)}")
            return faces  # Return unfiltered faces on failure
