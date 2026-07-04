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
    description=(
        "Real-time face detection (ResNet-SSD DNN + Haar fallback) exposed over HTTP. "
        "Uploaded images are processed **in memory only** and are not persisted. "
        "This is a free, open-source demonstration / trial project — not a certified "
        "commercial biometric platform. Use it only with images you own or are "
        "authorized to process, and only where any legally required notice/consent "
        "is in place."
    ),
    version="1.0.0",
)

# Image uploads are decoded, analyzed, and discarded within the request; nothing
# is written to disk. This header advertises that retention posture on every
# response (handy for clients and audits).
_DATA_RETENTION_NOTICE = "no image data stored; processed in-memory only"

# Cap the accepted upload size so a single huge request can't exhaust memory
# (the image is decoded into RAM). Default 10 MiB; override via MAX_UPLOAD_BYTES.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))


@app.middleware("http")
async def add_data_retention_header(request, call_next):
    """Stamp every response with the in-memory data-retention notice."""
    response = await call_next(request)
    response.headers["X-Data-Retention"] = _DATA_RETENTION_NOTICE
    return response


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
    """Detect faces in an uploaded image and return their bounding boxes as JSON.

    The uploaded image is processed in memory and discarded once the response is
    built — it is never written to disk. Use this endpoint only with images you
    own or are authorized to process, and only where any legally required
    notice/consent is in place.
    """
    # Read at most one byte past the cap so an oversized upload is rejected
    # without pulling the whole payload into memory.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes",
        )

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    faces = apply_nms(get_detector().detect_faces(frame, max_faces=max_faces))
    return {"count": len(faces), "faces": [_serialize(f) for f in faces]}
