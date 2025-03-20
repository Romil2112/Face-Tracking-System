"""
Visualization Module for Face Tracking
Author: Romil V. Shah
This module handles the visualization of face tracking results.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple

class TrackingVisualizer:
    """
    Class for visualizing face tracking results.
    """
    
    def __init__(self, rect_color: Tuple[int, int, int] = (0, 255, 0), 
                 rect_thickness: int = 2,
                 center_color: Tuple[int, int, int] = (0, 0, 255),
                 center_radius: int = 5,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale: float = 0.7,
                 font_color: Tuple[int, int, int] = (255, 255, 255),
                 font_thickness: int = 2):
        """
        Initialize the tracking visualizer with the given parameters.
        
        Args:
            rect_color: Color of the face rectangle (BGR)
            rect_thickness: Thickness of the face rectangle
            center_color: Color of the face center point (BGR)
            center_radius: Radius of the face center point
            font: Font to use for text
            font_scale: Scale of the font
            font_color: Color of the text (BGR)
            font_thickness: Thickness of the text
        """
        self.rect_color = rect_color
        self.rect_thickness = rect_thickness
        self.center_color = center_color
        self.center_radius = center_radius
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.fps = None
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict[str, any]]) -> np.ndarray:
        """
        Draw face tracking visualization on the frame.
        
        Args:
            frame: Input image frame
            faces: List of face information dictionaries
            
        Returns:
            Frame with visualization drawn
        """
        # Make a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw each detected face
        for i, face in enumerate(faces):
            x, y, w, h = face['rect']
            center_x, center_y = face['center']
            
            # Draw face rectangle
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), self.rect_color, self.rect_thickness)
            
            # Draw face center point
            cv2.circle(vis_frame, (center_x, center_y), self.center_radius, self.center_color, -1)
            
            # Draw face index and coordinates
            text = f"Face #{i+1}: ({center_x}, {center_y})"
            cv2.putText(vis_frame, text, (x, y - 10), self.font, self.font_scale, 
                        self.font_color, self.font_thickness)
        
        # Draw the number of faces detected
        faces_text = f"Faces detected: {len(faces)}"
        cv2.putText(vis_frame, faces_text, (10, 30), self.font, self.font_scale, 
                    self.font_color, self.font_thickness)
        
        # Draw FPS if provided
        if self.fps is not None:
            fps_text = f"FPS: {self.fps:.2f}"
            cv2.putText(vis_frame, fps_text, (10, 60), self.font, self.font_scale, 
                        self.font_color, self.font_thickness)
        
        return vis_frame
    
    def set_fps(self, fps: float):
        """
        Set the FPS to display.
        
        Args:
            fps: Frames per second value
        """
        self.fps = fps
