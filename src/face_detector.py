"""
Face Detection Module
Author: Romil V. Shah
This module handles face detection using Haar Cascade classifier.
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
        """
        self.face_cascade = cv2.CascadeClassifier()
        
        if os.path.isfile(cascade_path):
            success = self.face_cascade.load(cascade_path)
            if not success:
                self._load_from_opencv_data()
        else:
            self._load_from_opencv_data(cascade_path)
            
        if self.face_cascade.empty():
            raise ValueError("Error loading cascade classifier. Could not find a valid cascade file.")
        
        self.eye_cascade = cv2.CascadeClassifier()
        try:
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            success = self.eye_cascade.load(eye_cascade_path)
            if not success:
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
        """
        try:
            cascade_path = cv2.data.haarcascades + filename
            if not cascade_path.endswith(".xml"):
                cascade_path += ".xml"
            
            success = self.face_cascade.load(cascade_path)
            if not success:
                cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', filename)
                if not cascade_path.endswith(".xml"):
                    cascade_path += ".xml"
                success = self.face_cascade.load(cascade_path)
        except AttributeError:
            cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades', filename)
            if not cascade_path.endswith(".xml"):
                cascade_path += ".xml"
            success = self.face_cascade.load(cascade_path)
    
    def _verify_face_geometry(self, face_rect):
        """
        Verify if the face has reasonable geometric properties.
        """
        x, y, w, h = face_rect
        
        aspect_ratio = w / h
        if aspect_ratio < config.FACE_ASPECT_RATIO_MIN or aspect_ratio > config.FACE_ASPECT_RATIO_MAX:
            if config.DEBUG_MODE:
                print(f"Face rejected: aspect ratio {aspect_ratio:.2f} outside range [{config.FACE_ASPECT_RATIO_MIN}, {config.FACE_ASPECT_RATIO_MAX}]")
            return False
            
        if w < self.min_size[0] or h < self.min_size[1]:
            if config.DEBUG_MODE:
                print(f"Face rejected: size {w}x{h} smaller than minimum {self.min_size[0]}x{self.min_size[1]}")
            return False
            
        return True
    
    def calculate_confidence(self, face_info, eye_count):
        """
        Calculate a confidence score for face detection.
        """
        confidence = 0.5  # Base confidence
        
        if eye_count > 0:
            confidence += 0.3
        if eye_count > 1:
            confidence += 0.1
            
        x, y, w, h = face_info['rect']
        aspect_ratio = w / h
        if config.FACE_ASPECT_RATIO_MIN <= aspect_ratio <= config.FACE_ASPECT_RATIO_MAX:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def detect_faces(self, frame: np.ndarray, max_faces: int = None) -> List[Dict[str, any]]:
        """
        Detect faces in the given frame.
        """
        if config.DEBUG_MODE:
            print("Starting face detection...")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.face_cascade.empty():
            print("Warning: Face cascade is empty. Face detection will not work.")
            return []
        
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if config.DEBUG_MODE:
                print(f"Initial face detection found {len(faces)} faces")
                
        except Exception as e:
            print(f"Error during face detection: {e}")
            return []
        
        if len(faces) == 0:
            return []
            
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
        
        verified_face_info = []
        
        if not self.eye_cascade.empty() and config.EYE_DETECTION_ENABLED:
            for face in face_info:
                x, y, w, h = face['rect']
                
                if not self._verify_face_geometry(face['rect']):
                    if config.DEBUG_MODE:
                        print(f"Face at {face['rect']} rejected by geometry check")
                    continue
                
                roi_gray = gray[y:y+h, x:x+w]
                
                try:
                    eyes = self.eye_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=config.EYE_SCALE_FACTOR,
                        minNeighbors=config.EYE_MIN_NEIGHBORS,
                        minSize=config.EYE_MIN_SIZE
                    )
                    
                    confidence = self.calculate_confidence(face, len(eyes))
                    face['confidence'] = confidence
                    face['eye_count'] = len(eyes)
                    
                    if config.DEBUG_MODE:
                        print(f"Face at {face['rect']} has {len(eyes)} eyes, confidence: {confidence:.2f}")
                    
                    verified_face_info.append(face)
                    
                except Exception as e:
                    if config.DEBUG_MODE:
                        print(f"Error during eye detection: {e}")
                    face['confidence'] = 0.5
                    face['eye_count'] = 0
                    verified_face_info.append(face)
        else:
            for face in face_info:
                if self._verify_face_geometry(face['rect']):
                    face['confidence'] = 0.5
                    face['eye_count'] = 0
                    verified_face_info.append(face)
        
        if config.DEBUG_MODE:
            print(f"After verification: {len(verified_face_info)} faces remain")
            
        verified_face_info.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        if config.MINIMUM_CONFIDENCE > 0:
            verified_face_info = [f for f in verified_face_info if f['confidence'] >= config.MINIMUM_CONFIDENCE]
        
        if max_faces is not None and max_faces > 0:
            verified_face_info = verified_face_info[:max_faces]
            
        return verified_face_info
