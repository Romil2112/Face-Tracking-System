"""
Temporal Filtering Module for Face Tracking
Author: Romil V. Shah
This module implements temporal filtering to reduce false positives in face tracking.
"""

import numpy as np
from typing import List, Dict
from datetime import timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)

class TemporalFilter:
    def __init__(self, history_size: int = 5, consistency_threshold: int = 3):
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold
        self.face_history = deque(maxlen=history_size)

    def _validate_face(self, face: Dict) -> bool:
        required_keys = {'rect', 'center', 'confidence'}
        if not all(key in face for key in required_keys):
            logger.warning("Invalid face structure detected")
            return False
        x, y, w, h = face['rect']
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid face dimensions: {face['rect']}")
            return False
        return True

    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        try:
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2

            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area

            return inter_area / union_area if union_area > 0 else 0.0
        except Exception as e:
            logger.error(f"IoU calculation error: {str(e)}")
            return 0.0

    def _match_faces(self, current_faces: List[Dict], previous_faces: List[Dict]) -> List[int]:
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

    def _calculate_motion(self, faces: List[Dict], time_diff: timedelta) -> np.ndarray:
        """Calculate motion vector based on face positions and time difference."""
        if len(self.face_history) < 2:
            return np.zeros((len(faces), 2))

        prev_faces = self.face_history[-2]
        matches = self._match_faces(faces, prev_faces)
        motion_vector = np.zeros((len(faces), 2))
        
        # Add validation for time difference
        time_diff_seconds = max(time_diff.total_seconds(), 0.001)  # Prevent division by zero

        for i, match in enumerate(matches):
            if match != -1 and 0 <= match < len(prev_faces):
                try:
                    current_center = np.array(faces[i]['center'])
                    prev_center = np.array(prev_faces[match]['center'])
                    motion_vector[i] = (current_center - prev_center) / time_diff_seconds
                except (IndexError, KeyError) as e:
                    logger.warning(f"Invalid face index in motion calculation: {str(e)}")
                    motion_vector[i] = np.zeros(2)

        return motion_vector

    def _apply_motion_prediction(self, faces: List[Dict], motion_vector: np.ndarray, time_diff: timedelta) -> List[Dict]:
        """Apply motion prediction to face positions."""
        time_diff_seconds = max(time_diff.total_seconds(), 0.001)  # Prevent invalid multiplication
        
        for i, face in enumerate(faces):
            try:
                if np.isnan(motion_vector[i]).any():
                    continue
                    
                predicted_center = np.array(face['center']) + motion_vector[i] * time_diff_seconds
                if not np.isnan(predicted_center).any():
                    face['predicted_center'] = tuple(map(int, predicted_center))
            except (IndexError, KeyError) as e:
                logger.warning(f"Invalid motion prediction: {str(e)}")
        
        return faces

    def update(self, faces: List[Dict], time_diff: timedelta) -> List[Dict]:
        try:
            if not isinstance(faces, list):
                logger.error("Invalid input type for faces")
                return []

            valid_faces = [face for face in faces if self._validate_face(face)]
            self.face_history.append(valid_faces)

            if len(self.face_history) >= 2:
                motion_vector = self._calculate_motion(valid_faces, time_diff)
                valid_faces = self._apply_motion_prediction(valid_faces, motion_vector, time_diff)

            for face in valid_faces:
                if not self.face_history:
                    face['consistency'] = 1
                    continue

                matches = self._match_faces([face], self.face_history[-1])
                if matches and matches[0] != -1:
                    face['consistency'] = self.face_history[-1][matches[0]].get('consistency', 1) + 1
                else:
                    face['consistency'] = 1

            filtered_faces = [
                face for face in valid_faces
                if face.get('consistency', 0) >= self.consistency_threshold
            ]

            return filtered_faces
        except Exception as e:
            logger.error(f"Temporal filtering error: {str(e)}")
            return faces

    def reset(self):
        """Reset temporal filter state"""
        self.face_history.clear()
