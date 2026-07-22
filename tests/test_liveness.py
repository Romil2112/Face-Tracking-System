"""Tests for liveness detection integration."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import liveness
from liveness import LivenessDetector

client = TestClient(api.app)


def _png_bytes(shape=(60, 60, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _blank_frame(shape=(60, 60, 3)):
    return np.zeros(shape, dtype=np.uint8)


def test_liveness_absent_by_default():
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    for face in r.json()["faces"]:
        assert "liveness" not in face


def test_liveness_present_when_enabled(monkeypatch):
    detector = LivenessDetector()
    monkeypatch.setattr(api, "_liveness_detector", detector)
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    for face in r.json()["faces"]:
        assert face["liveness"]["checked"] is False
        assert face["liveness"]["reason"] == "single_frame_input"


def test_liveness_graceful_mediapipe_unavailable(monkeypatch):
    monkeypatch.setattr(liveness, "_MEDIAPIPE_AVAILABLE", False)
    frame = _blank_frame()
    result = LivenessDetector().check_frame_sequence([frame] * 5)
    assert result["checked"] is False
    assert result["reason"] == "mediapipe_unavailable"


def test_liveness_insufficient_frames():
    result = LivenessDetector().check_frame_sequence([])
    assert result["checked"] is False

def test_liveness_check_single_image_always_false():
    frame = _blank_frame()
    result = LivenessDetector().check_single_image(frame)
    assert result["checked"] is False
    assert result["reason"] == "single_frame_input"
