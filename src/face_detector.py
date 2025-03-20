"""
Face detection module using Haar Cascades.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import config

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
        
        # Load eye cascade for face verification
        self.eye_cascade = cv2.CascadeClassifier()
        try:
            # Try using OpenCV's built-in data
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            success = self.eye_cascade.load(eye_cascade_path)
            if not success:
                # Fallback method
                eye_cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', 'haarcascade_eye.xml')
                self.eye_cascade.load(eye_cascade_path)
        except Exception as e:
            print(f"Warning: Could not load eye cascade classifier: {e}")
            print("Face verification using eye detection will be disabled.")
            
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
    
    def _verify_face_geometry(self, face_rect):
        """
        Verify if the face has reasonable geometric properties.
        """
        x, y, w, h = face_rect
        
        # Check aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < config.FACE_ASPECT_RATIO_MIN or aspect_ratio > config.FACE_ASPECT_RATIO_MAX:
            if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                print(f"Face rejected: aspect ratio {aspect_ratio:.2f} outside range [{config.FACE_ASPECT_RATIO_MIN}, {config.FACE_ASPECT_RATIO_MAX}]")
            return False
            
        # Check minimum face size
        if w < self.min_size[0] or h < self.min_size[1]:
            if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                print(f"Face rejected: size {w}x{h} smaller than minimum {self.min_size[0]}x{self.min_size[1]}")
            return False
            
        return True
    
    def calculate_confidence(self, face_info, eye_count):
        """
        Calculate a confidence score for face detection.
        """
        confidence = 0.5  # Base confidence
        
        # More eyes = higher confidence
        if eye_count > 0:
            confidence += 0.3
        if eye_count > 1:
            confidence += 0.1
            
        # Good aspect ratio increases confidence
        x, y, w, h = face_info['rect']
        aspect_ratio = w / h
        if hasattr(config, 'FACE_ASPECT_RATIO_MIN') and hasattr(config, 'FACE_ASPECT_RATIO_MAX'):
            if config.FACE_ASPECT_RATIO_MIN <= aspect_ratio <= config.FACE_ASPECT_RATIO_MAX:
                confidence += 0.1
            
        return min(confidence, 1.0)
    
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
            - 'confidence': Confidence score for the detection
        """
        # Debugging
        if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
            print("Starting face detection...")
        
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
            
            if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                print(f"Initial face detection found {len(faces)} faces")
                
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
                'area': area,
                'confidence': 0.5  # Default confidence
            })
        
        # Add verification and confidence scoring
        verified_face_info = []
        
        if not self.eye_cascade.empty() and hasattr(config, 'EYE_DETECTION_ENABLED') and config.EYE_DETECTION_ENABLED:
            for face in face_info:
                x, y, w, h = face['rect']
                
                # Check face geometry
                geometry_valid = self._verify_face_geometry(face['rect'])
                if not geometry_valid and hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                    print(f"Face at {face['rect']} rejected by geometry check")
                    continue
                
                # Define region of interest for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detect eyes in the face region
                try:
                    eye
                except Exception as e:
                    if hasattr(config, 'DEBUG_MODE') and config.DEBUG_MODE:
                        print(f"Error during eye detection: {e}")
                    # If eye detection fails, still accept the face with base confidence
                    face['confidence'] = 0.5
                    face['eye_count'] = 0
                    verified_face_info.append(face)
