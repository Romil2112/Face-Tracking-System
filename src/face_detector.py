"""
Face detection module using Haar Cascades.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional

class FaceDetector:
    """
    Class for detecting faces in images using Haar Cascade classifier.
    """
    
    def __init__(self, cascade_path: str, scale_factor: float = 1.1, 
                 min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize the face detector with the given parameters.
        
        Args:
            cascade_path: Path to the Haar cascade XML file
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum size of the face to detect
        """
        # Try to use the provided path first
        self.face_cascade = cv2.CascadeClassifier()
        
        # Check if the file exists at the provided path
        if os.path.isfile(cascade_path):
            success = self.face_cascade.load(cascade_path)
            if not success:
                # If loading fails, try to use the OpenCV built-in path
                self._load_from_opencv_data()
        else:
            # If file doesn't exist, try to use the OpenCV built-in path
            self._load_from_opencv_data(cascade_path)
            
        # Check if cascade loaded successfully
        if self.face_cascade.empty():
            raise ValueError(f"Error loading cascade classifier. Could not find a valid cascade file.")
            
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
    
    def _load_from_opencv_data(self, filename="haarcascade_frontalface_default.xml"):
        """
        Attempts to load the cascade file from OpenCV's data directory.
        
        Args:
            filename: Name of the cascade file to load
        """
        try:
            # Try using cv2.data.haarcascades path (OpenCV 3.3+)
            cascade_path = cv2.data.haarcascades + filename
            if not cascade_path.endswith(".xml"):
                cascade_path += ".xml"
            
            success = self.face_cascade.load(cascade_path)
            if not success:
                # If still fails, try with dirname method
                cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', filename)
                if not cascade_path.endswith(".xml"):
                    cascade_path += ".xml"
                success = self.face_cascade.load(cascade_path)
        except AttributeError:
            # Fallback for older OpenCV versions
            cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', filename)
            if not cascade_path.endswith(".xml"):
                cascade_path += ".xml"
            success = self.face_cascade.load(cascade_path)
    
    def detect_faces(self, frame: np.ndarray, max_faces: int = None) -> List[Dict[str, any]]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image frame
            max_faces: Maximum number of faces to return (largest ones prioritized)
            
        Returns:
            List of dictionaries containing face information:
            - 'rect': (x, y, w, h) - Face rectangle coordinates
            - 'center': (cx, cy) - Center point of the face
            - 'area': Area of the face rectangle
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if cascade is loaded
        if self.face_cascade.empty():
            print("Warning: Face cascade is empty. Face detection will not work.")
            return []
        
        # Perform face detection
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except Exception as e:
            print(f"Error during face detection: {e}")
            return []
        
        # Handle case where no faces are detected but no exception is raised
        if len(faces) == 0:
            return []
            
        # Create face information list
        face_info = []
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            
            face_info.append({
                'rect': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area
            })
        
        # Sort faces by area (largest first) and limit if requested
        face_info.sort(key=lambda x: x['area'], reverse=True)
        if max_faces is not None and max_faces > 0 and len(face_info) > max_faces:
            face_info = face_info[:max_faces]
            
        return face_info
