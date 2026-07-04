"""Tests for FaceDetector using the real model/cascade files (headless)."""
import numpy as np
import pytest

import config
import face_detector
from face_detector import FaceDetector


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


@pytest.fixture
def blank():
    return np.zeros((120, 120, 3), dtype=np.uint8)


def test_constructs_with_real_files(detector):
    assert detector.dnn_initialized is True
    assert detector.haar_initialized is True


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


def test_detect_faces_dnn_blank_returns_list(detector, blank):
    faces = detector.detect_faces_dnn(blank)
    assert isinstance(faces, list)


def test_detect_faces_dnn_invalid_frame_returns_empty(detector):
    assert detector.detect_faces_dnn(None) == []


def test_detect_faces_haar_blank_returns_list(detector, blank):
    faces = detector.detect_faces_haar(blank)
    assert isinstance(faces, list)


def test_detect_faces_haar_invalid_frame_returns_empty(detector):
    assert detector.detect_faces_haar(None) == []


def test_detect_faces_pipeline(detector, blank):
    faces = detector.detect_faces(blank, max_faces=5)
    assert isinstance(faces, list)


def test_process_dnn_detections_filters_low_confidence(detector):
    # Build a synthetic (1,1,N,7) detections tensor.
    det = np.zeros((1, 1, 2, 7), dtype=np.float32)
    # Detection 0: high confidence, valid box.
    det[0, 0, 0] = [0, 1, 0.99, 0.1, 0.1, 0.5, 0.5]
    # Detection 1: below threshold -> dropped.
    det[0, 0, 1] = [0, 1, 0.10, 0.1, 0.1, 0.5, 0.5]
    faces = detector._process_dnn_detections(det, (100, 100))
    assert len(faces) == 1
    assert faces[0]["confidence"] == pytest.approx(0.99, abs=1e-4)
    assert faces[0]["rect"][2] > 0 and faces[0]["rect"][3] > 0


def test_process_dnn_detections_skips_degenerate_box(detector):
    det = np.zeros((1, 1, 1, 7), dtype=np.float32)
    # x1 == x2 and y1 == y2 -> w == h == 0 -> skipped.
    det[0, 0, 0] = [0, 1, 0.99, 0.5, 0.5, 0.5, 0.5]
    faces = detector._process_dnn_detections(det, (100, 100))
    assert faces == []


def test_detect_faces_dnn_handles_forward_error(blank, monkeypatch):
    d = FaceDetector()

    class BoomNet:
        def setInput(self, blob):
            pass

        def forward(self):
            raise RuntimeError("forward failed")

    # Replace the whole net object (its C attributes are read-only individually).
    d.net = BoomNet()
    # The except block disables DNN and returns [] (handle_face_detection_error
    # is the bound instance method here, so it works).
    assert d.detect_faces_dnn(blank) == []
    assert d.dnn_initialized is False


def test_detect_faces_haar_handles_error(monkeypatch, blank):
    d = FaceDetector()

    class BoomCascade:
        def detectMultiScale(self, *a, **k):
            raise RuntimeError("detectMultiScale failed")

    d.face_cascade = BoomCascade()
    assert d.detect_faces_haar(blank) == []
    assert d.haar_initialized is False


def test_valid_detections_drops_bad_rects():
    faces = [
        {"rect": (0, 0, 10, 10), "confidence": 0.9, "area": 100},
        {"rect": (0, 0, 0, 10), "confidence": 0.9, "area": 0},        # w <= 0
        {"rect": (0, 0, np.nan, 10), "confidence": 0.9, "area": 0},   # NaN
        {"confidence": 0.9},                                          # no rect (KeyError)
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
    # max_faces == 0 -> no cap applied; both kept.
    assert len(ranked) == 2


def test_detect_faces_uses_haar_fallback_when_dnn_empty(monkeypatch, blank):
    d = FaceDetector()
    monkeypatch.setattr(d, "detect_faces_dnn", lambda f: [])
    called = {"haar": False}

    def fake_haar(f):
        called["haar"] = True
        return []

    monkeypatch.setattr(d, "detect_faces_haar", fake_haar)
    d.detect_faces(blank)
    assert called["haar"] is True


def test_init_dnn_missing_files_raises():
    # Directly exercise the FileNotFoundError branch of _init_dnn via a detector
    # instance (bypassing __init__'s try/except by calling the method).
    d = FaceDetector()
    with pytest.raises(FileNotFoundError):
        d._init_dnn("/no/such/model", "/no/such/config")


def test_init_haar_missing_file_raises():
    d = FaceDetector()
    with pytest.raises(FileNotFoundError):
        d._init_haar("/no/such/cascade.xml")


def test_all_detection_methods_failing_raises(monkeypatch):
    # Force both initializers to fail so the constructor raises RuntimeError.
    monkeypatch.setattr(FaceDetector, "_init_dnn",
                        lambda self, *a: (_ for _ in ()).throw(RuntimeError("dnn")))
    monkeypatch.setattr(FaceDetector, "_init_haar",
                        lambda self, *a: (_ for _ in ()).throw(RuntimeError("haar")))
    with pytest.raises(RuntimeError, match="All face detection methods failed"):
        FaceDetector()


def test_module_imports():
    assert hasattr(face_detector, "FaceDetector")
