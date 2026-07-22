"""Tests for per-IP rate limiting on POST /detect."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _png_bytes(shape=(60, 60, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture(autouse=True)
def reset_limiter_storage():
    """Clear in-memory rate-limit counters before and after each test."""
    from rate_limiter import limiter

    storage = getattr(limiter, "_storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
    yield
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()


def test_within_limit_succeeds(monkeypatch):
    """Requests under the configured limit return 200."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000/minute")
    r1 = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    r2 = client.post("/detect", files={"file": ("b.png", _png_bytes(), "image/png")})
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_exceeding_limit_returns_429(monkeypatch):
    """The (limit+1)th request within the window returns HTTP 429."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1/minute")
    r1 = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    r2 = client.post("/detect", files={"file": ("b.png", _png_bytes(), "image/png")})
    assert r1.status_code == 200
    assert r2.status_code == 429
    body = r2.json()
    assert body["error"] == "rate_limit_exceeded"
    assert isinstance(body["retry_after"], int)
    assert body["retry_after"] > 0
    assert r2.headers.get("Retry-After") is not None


def test_429_response_has_correct_keys(monkeypatch):
    """429 body contains exactly {error, retry_after}."""
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1/minute")
    client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    r = client.post("/detect", files={"file": ("b.png", _png_bytes(), "image/png")})
    assert r.status_code == 429
    assert set(r.json()) == {"error", "retry_after"}


def test_413_still_works_with_rate_limiter(monkeypatch):
    """Oversized uploads still return 413 regardless of rate limiter."""
    monkeypatch.setattr("api.MAX_UPLOAD_BYTES", 100)
    r = client.post("/detect", files={"file": ("big.png", b"x" * 500, "image/png")})
    assert r.status_code == 413
