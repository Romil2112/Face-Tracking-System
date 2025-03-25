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
        Args:
            rect_color: BGR color tuple for bounding boxes
            rect_thickness: Thickness of bounding box lines
            center_color: BGR color tuple for center points
            center_radius: Radius of center circles
            font: OpenCV font type
            font_scale: Font size scale factor
            font_color: BGR color tuple for text
            font_thickness: Thickness of text strokes
        """
        self.rect_color = rect_color
        self.rect_thickness = max(1, rect_thickness)
        self.center_color = center_color
        self.center_radius = max(1, center_radius)
        self.font = font
        self.font_scale = max(0.5, font_scale)
        self.font_color = font_color
        self.font_thickness = max(1, font_thickness)
        self._fps = 0.0

    def set_fps(self, fps: float) -> None:
        """Set current FPS value with validation."""
        self._fps = max(0.0, min(fps, 1000.0))

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

    def _draw_fps(self, frame: np.ndarray) -> None:
        """Draw FPS counter with validation."""
        try:
            if frame.size == 0 or len(frame.shape) != 3:
                return
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                        self.font, self.font_scale,
                        self.font_color, self.font_thickness)
        except Exception as e:
            logger.error(f"FPS drawing failed: {str(e)}")

    def draw_perf_stats(self, frame: np.ndarray, timings: Dict[str, float]) -> None:
        """Draw performance statistics."""
        try:
            cv2.putText(frame, f"DNN: {timings.get('dnn', 0):.1f}ms", (10, 60),
                        self.font, self.font_scale, self.font_color, self.font_thickness)
            cv2.putText(frame, f"Haar: {timings.get('haar', 0):.1f}ms", (10, 90),
                        self.font, self.font_scale, self.font_color, self.font_thickness)
        except Exception as e:
            logger.error(f"Performance stats drawing failed: {str(e)}")

    def draw_faces(self, frame: np.ndarray, faces: List[Dict], timings: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Draw face annotations on frame with error recovery.
        Args:
            frame: Input BGR frame
            faces: List of face dictionaries
            timings: Optional dictionary of performance timings
        Returns:
            Output frame with visualizations
        """
        try:
            if frame is None or frame.size == 0:
                logger.error("Received invalid frame for visualization")
                return np.zeros((100, 100, 3), dtype=np.uint8)

            # Create copy to prevent modifying original frame
            output_frame = frame.copy()

            # Draw all face annotations
            for face in faces:
                if 'rect' in face:
                    x, y, w, h = face['rect']
                    self._safe_draw_rect(output_frame, x, y, w, h)
                if 'center' in face:
                    self._safe_draw_center(output_frame, face['center'])

            # Draw system status
            self._draw_fps(output_frame)
            if timings:
                self.draw_perf_stats(output_frame, timings)

            return output_frame
        except Exception as e:
            logger.error(f"Face visualization failed: {str(e)}")
            return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
