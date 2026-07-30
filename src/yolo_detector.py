"""
YOLO Face Detector
Author: Romil V. Shah

YOLOv8-nano face detector via the ultralytics library.  Returns face dicts
with the same schema as FaceDetector (rect, center, confidence, area) so the
two are interchangeable in the detection pipeline.

Model weights are downloaded by ultralytics on first use and cached under
~/.config/Ultralytics/.  The Docker image pre-downloads them at build time so
container start-up requires no network I/O.
"""

import logging

import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

import acceleration
import config

logger = logging.getLogger(__name__)


class YOLOFaceDetector:
    """YOLOv8-nano face detector, device-selected via the project acceleration policy."""

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL_NAME,
        confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD,
    ):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed — pip install ultralytics")
        device = acceleration.select_torch_device()
        self.model = YOLO(model_name)
        self.model.to(device)
        self.confidence_threshold = confidence_threshold
        logger.info("YOLO detector initialised (model=%s, device=%s)", model_name, device)

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a BGR frame; return face dicts compatible with FaceDetector."""
        if frame is None or frame.size == 0:
            return []
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(float(v)) for v in box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                faces.append({
                    "rect": (x1, y1, w, h),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "confidence": float(box.conf[0]),
                    "area": w * h,
                })
        return faces
