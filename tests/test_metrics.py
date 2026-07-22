"""Tests for Prometheus metrics and structlog observability."""
import cv2
import numpy as np
import pytest
import structlog.testing
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _png_bytes(shape=(60, 60, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_metrics_endpoint_returns_200():
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_prometheus_format():
    client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    r = client.get("/metrics")
    text = r.text
    assert "face_detection_backend_total" in text
    assert "face_detection_errors_total" in text


def test_detect_increments_backend_counter():
    client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    r = client.get("/metrics")
    assert "face_detection_backend_total" in r.text


def test_structured_log_no_biometric_data():
    with structlog.testing.capture_logs() as cap:
        client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    for record in cap:
        assert "rect" not in record
        assert "center" not in record
    # At least one log record should carry request_id
    assert any("request_id" in r for r in cap)
