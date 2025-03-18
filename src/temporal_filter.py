"""
Temporal filtering module for reducing false positives in face tracking.
"""

import numpy as np
from typing import List, Dict, Tuple

class TemporalFilter:
    """
    Class for filtering face detections across multiple frames.
    """
    
    def __init__(self, history_size: int = 5, consistency_threshold: int = 3):
        """
        Initialize the temporal filter.
        
        Args:
            history_size: Number of frames to track history
            consistency_threshold: Minimum number of consistent detections required
        """
        self.history_size = history_size
        self.consistency_threshold = consistency_threshold
        self.face_history = []
        
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1, box2: Bounding boxes in format (x, y, w, h)
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to (x1, y1, x2, y2) format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    
    def _match_faces(self, faces: List[Dict], 
                    previous_faces: List[Dict], 
                    iou_threshold: float = 0.3) -> List[int]:
        """
        Match current faces with faces detected in the previous frame.
        
        Args:
            faces: Current frame face detections
            previous_faces: Previous frame face detections
            iou_threshold: Minimum IoU to consider a match
            
        Returns:
            List of indices matching current faces to previous faces (-1 if no match)
        """
        if not previous_faces or not faces:
            return [-1] * len(faces)
            
        matches = []
        for face in faces:
            best_match = -1
            best_iou = iou_threshold
            
            for i, prev_face in enumerate(previous_faces):
                iou = self._calculate_iou(face['rect'], prev_face['rect'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
                    
            matches.append(best_match)
        return matches
    
    def update(self, faces: List[Dict]) -> List[Dict]:
        """
        Update face history and filter faces based on temporal consistency.
        
        Args:
            faces: Current frame face detections
            
        Returns:
            Filtered face detections
        """
        if len(self.face_history) == 0:
            # First frame
            self.face_history.append(faces)
            return faces
            
        # Match current faces with previous faces
        matches = self._match_faces(faces, self.face_history[-1])
        
        # Update consistency counts
        for face, match_idx in zip(faces, matches):
            if match_idx >= 0:
                # This face matched a previous face, increment its consistency count
                face['consistency'] = self.face_history[-1][match_idx].get('consistency', 1) + 1
            else:
                # New face
                face['consistency'] = 1
                
        # Add current faces to history
        self.face_history.append(faces)
        
        # Remove oldest frame if history is too long
        if len(self.face_history) > self.history_size:
            self.face_history.pop(0)
            
        # Filter faces based on consistency threshold
        filtered_faces = [face for face in faces 
                         if face.get('consistency', 1) >= self.consistency_threshold]
        
        return filtered_faces
