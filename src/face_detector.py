"""
Face Detection Module
Author: Romil V. Shah
This module handles face detection using YOLO (primary) and Haar Cascade (fallback).
"""

import logging
import os

import cv2
import numpy as np

import config
from error_handling import ErrorHandler, retry
from yolo_detector import YOLO_AVAILABLE, YOLOFaceDetector

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector with YOLO primary and Haar Cascade last-resort fallback.
    Implements error recovery: if YOLO fails to initialize, detection continues
    via Haar Cascade alone.
    """

    def __init__(
        self,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
        cascade_path: str = config.CASCADE_PATH,
        scale_factor: float = config.SCALE_FACTOR,
        min_neighbors: int = config.MIN_NEIGHBORS,
        min_size: tuple[int, int] = config.MIN_SIZE,
    ):
        self._error_handler = ErrorHandler()
        self.yolo_initialized = False
        self.haar_initialized = False
        try:
            self._yolo = YOLOFaceDetector(confidence_threshold=confidence_threshold)
            self.yolo_initialized = True
        except Exception as e:
            logger.error("YOLO initialization failed: %s", e)
            self._error_handler.handle_face_detection_error(e)
        try:
            self._init_haar(cascade_path)
            self.haar_initialized = True
        except Exception as e:
            logger.error("Haar Cascade initialization failed: %s", e)
            self._error_handler.handle_face_detection_error(e)
        if not self.yolo_initialized and not self.haar_initialized:
            raise RuntimeError("All face detection methods failed to initialize")
        self.confidence_threshold = confidence_threshold
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

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

    def detect_faces_yolo(self, frame: np.ndarray) -> list[dict]:
        """YOLO-based face detection with error recovery."""
        if not self.yolo_initialized or not self._validate_frame(frame):
            return []
        try:
            return self._yolo.detect_faces(frame)
        except Exception as e:
            logger.error("YOLO detection failed: %s", e)
            self._error_handler.handle_face_detection_error(e)
            self.yolo_initialized = False
            return []

    def detect_faces_haar(self, frame: np.ndarray) -> list[dict]:
        """Haar Cascade face detection — last-resort fallback."""
        if not self.haar_initialized or not self._validate_frame(frame):
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            return [
                {
                    "rect": (x, y, w, h),
                    "center": (x + w // 2, y + h // 2),
                    "confidence": 0.5,
                    "area": w * h,
                }
                for (x, y, w, h) in faces
            ]
        except Exception as e:
            logger.error("Haar Cascade detection failed: %s", e)
            self._error_handler.handle_face_detection_error(e)
            self.haar_initialized = False
            return []

    def detect_faces(self, frame: np.ndarray, max_faces: int = None) -> list[dict]:
        """Main detection: YOLO primary, Haar last-resort fallback."""
        if not self._validate_frame(frame):
            return []
        faces = []
        if self.yolo_initialized:
            faces = self.detect_faces_yolo(frame)
        if not faces and config.DEBUG_MODE:
            logger.info("YOLO detected no faces, trying Haar Cascade")
        if not faces and self.haar_initialized:
            faces = self.detect_faces_haar(frame)
        valid_faces = self._valid_detections(faces)
        return self._rank_and_limit(valid_faces, max_faces)

    def backend_name(self) -> str:
        """String identifying the active inference backend, for metrics and logging."""
        if self.yolo_initialized:
            try:
                return self._yolo.model.device.type.upper()
            except Exception:
                return "CPU"
        return "CPU"

    @staticmethod
    def _valid_detections(faces: list[dict]) -> list[dict]:
        """Drop detections with a malformed/degenerate rect (NaN or non-positive)."""
        valid_faces = []
        for f in faces:
            try:
                x, y, w, h = f["rect"]
                if any(np.isnan([x, y, w, h])) or w <= 0 or h <= 0:
                    continue
                valid_faces.append(f)
            except (KeyError, TypeError):
                continue
        return valid_faces

    @staticmethod
    def _rank_and_limit(valid_faces: list[dict], max_faces: int) -> list[dict]:
        """Sort by (confidence, area) desc, drop low-confidence, cap at max_faces."""
        valid_faces.sort(key=lambda x: (x["confidence"], x["area"]), reverse=True)
        if config.MINIMUM_CONFIDENCE > 0:
            valid_faces = [f for f in valid_faces if f["confidence"] >= config.MINIMUM_CONFIDENCE]
        if max_faces and max_faces > 0:
            valid_faces = valid_faces[:max_faces]
        return valid_faces

    def __del__(self):
        if hasattr(self, "face_cascade"):
            del self.face_cascade
