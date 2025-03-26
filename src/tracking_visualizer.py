"""
Tracking Visualization Module
Author: Romil V. Shah
This module handles visualization of face tracking results with robust error handling.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import config
import logging

logger = logging.getLogger(__name__)

class TrackingVisualizer:
    """
    Class for visualizing face tracking results with validation and error recovery.
    """
    
    def __init__(self,
                 rect_color: Tuple[int, int, int] = config.FACE_RECT_COLOR,
                 rect_thickness: int = config.FACE_RECT_THICKNESS,
                 center_color: Tuple[int, int, int] = config.FACE_CENTER_COLOR,
                 center_radius: int = config.FACE_CENTER_RADIUS,
                 font: int = getattr(cv2, config.FONT),
                 font_scale: float = config.FONT_SCALE,
                 font_color: Tuple[int, int, int] = config.FONT_COLOR,
                 font_thickness: int = config.FONT_THICKNESS):
        """
        Initialize visualizer with validation.
        """
        self.rect_color = rect_color
        self.rect_thickness = max(1, rect_thickness)
        self.center_color = center_color
        self.center_radius = max(1, center_radius)
        self.font = font
        self.font_scale = max(0.5, font_scale)
        self.font_color = font_color
        self.font_thickness = max(1, font_thickness)

    def _safe_draw_rect(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Draw rectangle with bounds checking."""
        try:
            height, width = frame.shape[:2]
            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(0, min(x + w, width - 1))
            y2 = max(0, min(y + h, height - 1))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            self.rect_color, self.rect_thickness)
        except Exception as e:
            logger.error(f"Rectangle drawing failed: {str(e)}")

    def _safe_draw_center(self, frame: np.ndarray, center: Tuple[int, int]) -> None:
        """Draw center point with bounds checking."""
        try:
            height, width = frame.shape[:2]
            cx = max(0, min(center[0], width - 1))
            cy = max(0, min(center[1], height - 1))
            cv2.circle(frame, (cx, cy), self.center_radius,
                      self.center_color, -1)
        except Exception as e:
            logger.error(f"Center drawing failed: {str(e)}")

    def draw_faces(self, frame: np.ndarray, faces: List[Dict], fps: float) -> np.ndarray:
        """
        Draw face annotations and system metrics on frame with error recovery.
        
        Args:
            frame: Input BGR frame
            faces: List of face dictionaries
            fps: Current frames per second
            
        Returns:
            Output frame with visualizations
        """
        try:
            if frame is None or frame.size == 0:
                logger.error("Received invalid frame for visualization")
                return np.zeros((100, 100, 3), dtype=np.uint8)

            output_frame = frame.copy()

            # Draw face annotations
            for face in faces:
                try:
                    x, y, w, h = face['rect']
                    if any(np.isnan([x, y, w, h])):
                        continue
                    self._safe_draw_rect(output_frame, x, y, w, h)
                    
                    # Draw confidence score
                    if 'confidence' in face:
                        confidence_text = f"Conf: {face['confidence']:.2f}"
                        cv2.putText(output_frame, confidence_text, 
                                   (x, y - 10), self.font,
                                   self.font_scale, self.font_color,
                                   self.font_thickness)

                    if 'center' in face:
                        self._safe_draw_center(output_frame, face['center'])
                except (KeyError, TypeError):
                    continue

            # Draw system metrics
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(output_frame, fps_text, (10, 30),
                       self.font, self.font_scale,
                       self.font_color, self.font_thickness)

            return output_frame

        except Exception as e:
            logger.error(f"Face visualization failed: {str(e)}")
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
