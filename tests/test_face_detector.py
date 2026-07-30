"""Tests for FaceDetector using the real cascade file (headless).

YOLO is the primary detector and Haar Cascade is the last-resort fallback.
"""
import numpy as np
import pytest

import config
import face_detector
from face_detector import FaceDetector
from yolo_detector import YOLO_AVAILABLE


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


@pytest.fixture
def blank():
    return np.zeros((120, 120, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructs_with_haar_always(detector):
    assert detector.haar_initialized is True


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_constructs_with_yolo_when_available(detector):
    assert detector.yolo_initialized is True


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------

def test_validate_frame_accepts_bgr(detector, blank):
    assert detector._validate_frame(blank) is True


def test_validate_frame_rejects_none(detector):
    assert detector._validate_frame(None) is False


def test_validate_frame_rejects_empty(detector):
    assert detector._validate_frame(np.zeros((0, 0, 3), dtype=np.uint8)) is False


def test_validate_frame_rejects_grayscale(detector):
    assert detector._validate_frame(np.zeros((10, 10), dtype=np.uint8)) is False


def test_validate_frame_rejects_wrong_channels(detector):
    assert detector._validate_frame(np.zeros((10, 10, 4), dtype=np.uint8)) is False


# ---------------------------------------------------------------------------
# YOLO detection path
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_yolo_blank_returns_list(detector, blank):
    faces = detector.detect_faces_yolo(blank)
    assert isinstance(faces, list)


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_yolo_invalid_frame_returns_empty(detector):
    assert detector.detect_faces_yolo(None) == []


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_yolo_handles_error(blank, monkeypatch):
    d = FaceDetector()

    def boom(frame):
        raise RuntimeError("YOLO boom")

    monkeypatch.setattr(d._yolo, "detect_faces", boom)
    assert d.detect_faces_yolo(blank) == []
    assert d.yolo_initialized is False


# ---------------------------------------------------------------------------
# Haar detection path
# ---------------------------------------------------------------------------

def test_detect_faces_haar_blank_returns_list(detector, blank):
    faces = detector.detect_faces_haar(blank)
    assert isinstance(faces, list)


def test_detect_faces_haar_invalid_frame_returns_empty(detector):
    assert detector.detect_faces_haar(None) == []


def test_detect_faces_haar_handles_error(monkeypatch, blank):
    d = FaceDetector()

    class BoomCascade:
        def detectMultiScale(self, *a, **k):
            raise RuntimeError("detectMultiScale failed")

    d.face_cascade = BoomCascade()
    assert d.detect_faces_haar(blank) == []
    assert d.haar_initialized is False


# ---------------------------------------------------------------------------
# Primary pipeline + fallback
# ---------------------------------------------------------------------------

def test_detect_faces_pipeline(detector, blank):
    faces = detector.detect_faces(blank, max_faces=5)
    assert isinstance(faces, list)


def test_detect_faces_uses_haar_fallback_when_yolo_empty(monkeypatch, blank):
    d = FaceDetector()
    monkeypatch.setattr(d, "detect_faces_yolo", lambda f: [])
    called = {"haar": False}

    def fake_haar(f):
        called["haar"] = True
        return []

    monkeypatch.setattr(d, "detect_faces_haar", fake_haar)
    d.detect_faces(blank)
    assert called["haar"] is True


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

def test_valid_detections_drops_bad_rects():
    faces = [
        {"rect": (0, 0, 10, 10), "confidence": 0.9, "area": 100},
        {"rect": (0, 0, 0, 10), "confidence": 0.9, "area": 0},        # w <= 0
        {"rect": (0, 0, np.nan, 10), "confidence": 0.9, "area": 0},   # NaN
        {"confidence": 0.9},                                           # no rect (KeyError)
    ]
    valid = FaceDetector._valid_detections(faces)
    assert len(valid) == 1
    assert valid[0]["rect"] == (0, 0, 10, 10)


def test_rank_and_limit_sorts_and_caps(monkeypatch):
    monkeypatch.setattr(config, "MINIMUM_CONFIDENCE", 0.4)
    faces = [
        {"rect": (0, 0, 10, 10), "confidence": 0.5, "area": 100},
        {"rect": (0, 0, 10, 10), "confidence": 0.9, "area": 200},
        {"rect": (0, 0, 10, 10), "confidence": 0.1, "area": 50},  # below min conf
    ]
    ranked = FaceDetector._rank_and_limit(faces, max_faces=1)
    assert len(ranked) == 1
    assert ranked[0]["confidence"] == 0.9


def test_rank_and_limit_no_min_conf(monkeypatch):
    monkeypatch.setattr(config, "MINIMUM_CONFIDENCE", 0)
    faces = [
        {"rect": (0, 0, 10, 10), "confidence": 0.1, "area": 50},
        {"rect": (0, 0, 10, 10), "confidence": 0.9, "area": 200},
    ]
    ranked = FaceDetector._rank_and_limit(faces, max_faces=0)
    # max_faces == 0 → no cap applied; both kept.
    assert len(ranked) == 2


# ---------------------------------------------------------------------------
# Initialization error paths
# ---------------------------------------------------------------------------

def test_init_haar_missing_file_raises():
    d = FaceDetector()
    with pytest.raises(FileNotFoundError):
        d._init_haar("/no/such/cascade.xml")


def test_all_detection_methods_failing_raises(monkeypatch):
    monkeypatch.setattr(FaceDetector, "_init_haar",
                        lambda self, *a: (_ for _ in ()).throw(RuntimeError("haar")))
    # Also prevent YOLO from initializing.
    import yolo_detector as _yd
    monkeypatch.setattr(_yd, "YOLO_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="All face detection methods failed"):
        FaceDetector()


# ---------------------------------------------------------------------------
# Module-level smoke
# ---------------------------------------------------------------------------

def test_module_imports():
    assert hasattr(face_detector, "FaceDetector")
