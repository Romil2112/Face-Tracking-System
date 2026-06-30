"""Tests for the FastAPI face-detection service (no camera required)."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _png_bytes(shape=(120, 120, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_returns_expected_structure():
    resp = client.post("/detect", files={"file": ("blank.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"count", "faces"}
    assert isinstance(body["faces"], list)
    assert body["count"] == len(body["faces"])


def test_detect_rejects_non_image_payload():
    resp = client.post("/detect", files={"file": ("bad.txt", b"not-an-image", "text/plain")})
    assert resp.status_code == 400


def test_detect_rejects_empty_file():
    resp = client.post("/detect", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_data_retention_header_present():
    resp = client.post("/detect", files={"file": ("blank.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert resp.headers["X-Data-Retention"] == "no image data stored; processed in-memory only"
