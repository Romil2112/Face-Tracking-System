"""Tests for Claude triage integration on POST /detect?triage=true."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import api
import triage
from triage import triage_detection

client = TestClient(api.app)


def _png_bytes(shape=(60, 60, 3)):
    ok, buf = cv2.imencode(".png", np.zeros(shape, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_triage_absent_by_default():
    r = client.post("/detect", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    for face in r.json()["faces"]:
        assert "triage_note" not in face


def test_triage_graceful_no_client(monkeypatch):
    monkeypatch.setattr(triage, "_client", None)
    r = client.post(
        "/detect?triage=true",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    for face in r.json()["faces"]:
        assert "triage_note" not in face


def test_triage_note_present_low_confidence(monkeypatch):
    fake_text = MagicMock()
    fake_text.text = "Check lighting."
    fake_msg = MagicMock()
    fake_msg.content = [fake_text]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_msg

    monkeypatch.setattr(triage, "_client", fake_client)
    monkeypatch.setattr(triage, "TRIAGE_CONFIDENCE_THRESHOLD", 1.0)

    r = client.post(
        "/detect?triage=true",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    # All faces are below threshold=1.0, so each should have triage_note
    for face in r.json()["faces"]:
        assert face.get("triage_note") == "Check lighting."


def test_triage_skips_high_confidence(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(triage, "_client", fake_client)
    monkeypatch.setattr(triage, "TRIAGE_CONFIDENCE_THRESHOLD", 0.0)

    r = client.post(
        "/detect?triage=true",
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    fake_client.messages.create.assert_not_called()
    for face in r.json()["faces"]:
        assert "triage_note" not in face
