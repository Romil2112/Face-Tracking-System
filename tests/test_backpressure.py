"""Tests for integer-counter backpressure on POST /detect."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def _png_bytes(shape=(60, 60, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_under_capacity_succeeds():
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200


def test_over_capacity_returns_503(monkeypatch):
    monkeypatch.setattr(api, "_detection_slots", 0)
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "server_busy"
    assert isinstance(body["retry_after"], int)
    assert body["retry_after"] > 0
    assert r.headers.get("Retry-After") is not None


def test_slot_restored_on_exception(monkeypatch):
    """Slot counter is restored even when the detector raises."""
    monkeypatch.setattr(api, "_detection_slots", 3)

    def boom(*args, **kwargs):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(api, "get_detector", boom)

    safe_client = TestClient(api.app, raise_server_exceptions=False)
    safe_client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert api._detection_slots == 3
