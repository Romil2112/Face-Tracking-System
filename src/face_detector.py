"""
Face Detection Module
Author: Romil V. Shah
This module handles face detection using both DNN and Haar Cascade methods with error recovery.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional
import config
from error_handling import retry, ErrorHandler

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Robust face detector with DNN and Haar Cascade fallback.
    Implements error recovery and hardware acceleration.
    """

    def __init__(self, dnn_model_path: str = config.DNN_MODEL_PATH,
                 dnn_config_path: str = config.DNN_CONFIG_PATH,
                 confidence_threshold: float = config.DNN_CONFIDENCE_THRESHOLD,
                 input_size: Tuple[int, int] = config.DNN_INPUT_SIZE,
                 cascade_path: str = config.CASCADE_PATH,
                 scale_factor: float = config.SCALE_FACTOR,
                 min_neighbors: int = config.MIN_NEIGHBORS,
                 min_size: Tuple[int, int] = config.MIN_SIZE):
        """Initialize detector with error recovery mechanisms."""
        self.dnn_initialized = False
        self.haar_initialized = False
        try:
            self._init_dnn(dnn_model_path, dnn_config_path)
            self.dnn_initialized = True
        except Exception as e:
            logger.error(f"DNN initialization failed: {e}")
            ErrorHandler.handle_face_detection_error(e)
        try:
            self._init_haar(cascade_path)
            self.haar_initialized = True
        except Exception as e:
            logger.error(f"Haar Cascade initialization failed: {e}")
            ErrorHandler.handle_face_detection_error(e)
        if not self.dnn_initialized and not self.haar_initialized:
            raise RuntimeError("All face detection methods failed to initialize")
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    @retry(max_attempts=3, delay=1, allowed_exceptions=(Exception,))
    def _init_dnn(self, model_path: str, config_path: str):
        """Initialize DNN detector with proper backend validation"""
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"DNN model files missing: {model_path}, {config_path}")
        self.net = cv2.dnn.readNet(model_path, config_path)
        # Get available backends and targets
        available_backends = getattr(cv2.dnn, 'getAvailableBackends', lambda: [])()
        if cv2.dnn.DNN_BACKEND_CUDA in available_backends:
            cuda_targets = getattr(cv2.dnn, 'getAvailableTargets', lambda x: [])(cv2.dnn.DNN_BACKEND_CUDA)
            if cv2.dnn.DNN_TARGET_CUDA in cuda_targets:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("CUDA acceleration enabled")
                return
        # Fallback to OpenCV CPU backend
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("Using CPU fallback")

    @retry(max_attempts=3, delay=1, allowed_exceptions=(Exception,))
    def _init_haar(self, cascade_path: str):
        """Initialize Haar Cascade classifier with retry logic."""
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar Cascade file missing: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Loaded empty Haar Cascade classifier")

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validate input frame dimensions and type."""
        if frame is None or frame.size == 0:
            logger.error("Received invalid frame (empty or None)")
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.error("Invalid frame format, expected 3-channel BGR image")
            return False
        return True

    def _process_dnn_detections(self, detections: np.ndarray, frame_shape: Tuple[int, int]) -> List[Dict]:
        """Process raw DNN detections into face dictionaries."""
        height, width = frame_shape
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            # Ensure bounding box coordinates are within frame bounds
            x1 = max(0, min(int(detections[0, 0, i, 3] * width), width - 1))
            y1 = max(0, min(int(detections[0, 0, i, 4] * height), height - 1))
            x2 = max(0, min(int(detections[0, 0, i, 5] * width), width - 1))
            y2 = max(0, min(int(detections[0, 0, i, 6] * height), height - 1))
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue  # Skip invalid boxes
            faces.append({
                'rect': (x1, y1, w, h),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'confidence': float(confidence),
                'area': w * h
            })
        return faces

    def detect_faces_dnn(self, frame: np.ndarray) -> List[Dict]:
        """Perform DNN-based face detection with error recovery."""
        if not self.dnn_initialized or not self._validate_frame(frame):
            return []
        try:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0,
                size=self.input_size,
                mean=(104.0, 177.0, 123.0),
                swapRB=False
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            return self._process_dnn_detections(detections, frame.shape[:2])
        except Exception as e:
            logger.error(f"DNN detection failed: {e}")
            ErrorHandler.handle_face_detection_error(e)
            self.dnn_initialized = False  # Disable DNN for subsequent frames
            return []

    def detect_faces_haar(self, frame: np.ndarray) -> List[Dict]:
        """Perform Haar Cascade-based face detection with fallback."""
        if not self.haar_initialized or not self._validate_frame(frame):
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return [{
                'rect': (x, y, w, h),
                'center': (x + w//2, y + h//2),
                'confidence': 0.5,
                'area': w * h
            } for (x, y, w, h) in faces]
        except Exception as e:
            logger.error(f"Haar Cascade detection failed: {e}")
            ErrorHandler.handle_face_detection_error(e)
            self.haar_initialized = False  # Disable Haar for subsequent frames
            return []

    def detect_faces(self, frame: np.ndarray, max_faces: int = None) -> List[Dict]:
        """Main detection method with automatic fallback and result validation."""
        if not self._validate_frame(frame):
            return []

        # Try DNN first if available
        faces = []
        if self.dnn_initialized:
            faces = self.detect_faces_dnn(frame)
        if not faces and config.DEBUG_MODE:
            logger.info("DNN detected no faces, trying Haar Cascade")

        # Fallback to Haar Cascade if needed
        if not faces and self.haar_initialized:
            faces = self.detect_faces_haar(frame)

        # Post-processing
        valid_faces = []
        for f in faces:
            try:
                x, y, w, h = f['rect']
                if any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0:
                    continue
                valid_faces.append(f)
            except (KeyError, TypeError):
                continue

        # Sort and limit results
        valid_faces.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        if config.MINIMUM_CONFIDENCE > 0:
            valid_faces = [f for f in valid_faces if f['confidence'] >= config.MINIMUM_CONFIDENCE]
        if max_faces and max_faces > 0:
            valid_faces = valid_faces[:max_faces]

        return valid_faces

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'face_cascade'):
            del self.face_cascade
