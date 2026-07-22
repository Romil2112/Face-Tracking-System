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

import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from face_detector import FaceDetector
from metrics import (
    build_instrumentator,
    configure_structlog,
    logger,
    new_request_id,
    record_backend,
    record_error,
)
from nms_utils import apply_nms
from liveness import LIVENESS_CHECK_ENABLED, LivenessDetector
from rate_limiter import get_rate_limit, limiter, rate_limit_exceeded_handler
from triage import TRIAGE_CONFIDENCE_THRESHOLD, triage_detection

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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

configure_structlog()
_instrumentator = build_instrumentator()
_instrumentator.instrument(app).expose(app)

MAX_CONCURRENT_DETECTIONS = int(os.environ.get("MAX_CONCURRENT_DETECTIONS", "10"))
# Plain integer counter — no asyncio.Semaphore needed for non-blocking backpressure.
# In asyncio there is no preemption between two non-awaited statements, so the
# check-then-decrement below is atomic within a single event loop turn.
_detection_slots: int = MAX_CONCURRENT_DETECTIONS

_liveness_detector: LivenessDetector | None = (
    LivenessDetector() if LIVENESS_CHECK_ENABLED else None
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
@limiter.limit(get_rate_limit)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    max_faces: int = Query(default=10, ge=1, le=100),
    triage: bool = Query(default=False),
) -> dict:
    """Detect faces in an uploaded image and return their bounding boxes as JSON.

    The uploaded image is processed in memory and discarded once the response is
    built — it is never written to disk. Use this endpoint only with images you
    own or are authorized to process, and only where any legally required
    notice/consent is in place.
    """
    request_id = new_request_id()
    t0 = time.monotonic()

    global _detection_slots
    _slot_taken = False
    try:
        if _detection_slots <= 0:
            record_error("server_busy")
            return JSONResponse(
                status_code=503,
                content={"error": "server_busy", "retry_after": 5},
                headers={"Retry-After": "5"},
            )
        _detection_slots -= 1
        _slot_taken = True

        # Read at most one byte past the cap so an oversized upload is rejected
        # without pulling the whole payload into memory.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            record_error("empty_file")
            raise HTTPException(status_code=400, detail="Empty file")
        if len(data) > MAX_UPLOAD_BYTES:
            record_error("oversized_upload")
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {MAX_UPLOAD_BYTES} bytes",
            )

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            record_error("decode_failure")
            raise HTTPException(status_code=400, detail="Could not decode image")

        detector = get_detector()
        faces = apply_nms(detector.detect_faces(frame, max_faces=max_faces))
        backend = detector.acceleration.name if hasattr(detector, "acceleration") else "unknown"

        serialized = [_serialize(f) for f in faces]
        if _liveness_detector is not None:
            for face in serialized:
                face["liveness"] = _liveness_detector.check_single_image(frame)

        if triage:
            for face in serialized:
                if face["confidence"] < TRIAGE_CONFIDENCE_THRESHOLD:
                    note = triage_detection(face)
                    if note is not None:
                        face["triage_note"] = note

        record_backend(backend)
        logger.info(
            "detection_complete",
            request_id=request_id,
            backend=backend,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            face_count=len(serialized),
            outcome="success",
        )

        return {"count": len(serialized), "faces": serialized}
    finally:
        if _slot_taken:
            _detection_slots += 1
