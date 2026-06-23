"""
Face Detection REST API
Author: Romil V. Shah

A headless HTTP service that wraps the FaceDetector (no camera required).
Run locally with:  uvicorn src.api:app --reload
"""
import os
import sys

# Allow the bare intra-package imports (`import config`) used across src/.
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from face_detector import FaceDetector
from nms_utils import apply_nms

app = FastAPI(
    title="Face Detection API",
    description="Real-time face detection (ResNet-SSD DNN + Haar fallback) exposed over HTTP.",
    version="1.0.0",
)

_detector = None


def get_detector() -> FaceDetector:
    """Lazily construct a single shared detector (loads models once)."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def _serialize(face: dict) -> dict:
    return {
        "rect": list(face["rect"]),
        "center": list(face["center"]),
        "confidence": round(float(face["confidence"]), 4),
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), max_faces: int = 10) -> dict:
    """Detect faces in an uploaded image and return their bounding boxes as JSON."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    faces = apply_nms(get_detector().detect_faces(frame, max_faces=max_faces))
    return {"count": len(faces), "faces": [_serialize(f) for f in faces]}
