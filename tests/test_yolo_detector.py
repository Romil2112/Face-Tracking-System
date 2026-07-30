"""Tests for YOLOFaceDetector.

YOLO inference is exercised via monkeypatched ultralytics objects so no model
weights need to be downloaded during test runs.  The YOLO_AVAILABLE flag is
checked so tests that exercise the live class are skipped automatically when
ultralytics is not installed.
"""
import numpy as np
import pytest

import yolo_detector
from yolo_detector import YOLO_AVAILABLE


# ---------------------------------------------------------------------------
# Minimal stand-ins for ultralytics result objects
# ---------------------------------------------------------------------------

class _FakeBox:
    """Mimics one bounding-box entry from ultralytics Results.boxes."""

    def __init__(self, coords, conf):
        # xyxy[0] must be iterable so (int(float(v)) for v in box.xyxy[0]) works.
        self.xyxy = [coords]
        self.conf = [float(conf)]


class _FakeResult:
    def __init__(self, boxes_data):
        self.boxes = [_FakeBox(c, conf) for c, conf in boxes_data]


class _FakeModel:
    """Stand-in for a YOLO model instance."""

    def __init__(self, boxes_data=None):
        self._boxes_data = boxes_data or []
        self.device = type("_D", (), {"type": "cpu"})()

    def to(self, device):
        return self

    def predict(self, source, conf=0.5, verbose=False):
        return [_FakeResult(self._boxes_data)]


def _patch_yolo(monkeypatch, boxes_data=None):
    """Replace ultralytics.YOLO in the module under test with a fake."""
    monkeypatch.setattr(yolo_detector, "YOLO", lambda name: _FakeModel(boxes_data))


# ---------------------------------------------------------------------------
# Flag / import tests (always run)
# ---------------------------------------------------------------------------

def test_yolo_available_flag_is_bool():
    assert isinstance(YOLO_AVAILABLE, bool)


def test_module_has_expected_names():
    assert hasattr(yolo_detector, "YOLO_AVAILABLE")
    assert hasattr(yolo_detector, "YOLOFaceDetector")


def test_raises_import_error_when_unavailable(monkeypatch):
    monkeypatch.setattr(yolo_detector, "YOLO_AVAILABLE", False)
    with pytest.raises(ImportError, match="ultralytics"):
        yolo_detector.YOLOFaceDetector()


# ---------------------------------------------------------------------------
# Behavioural tests (require ultralytics on PATH; skipped otherwise)
# ---------------------------------------------------------------------------

@pytest.fixture
def blank():
    return np.zeros((120, 120, 3), dtype=np.uint8)


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_none_frame_returns_empty(monkeypatch):
    _patch_yolo(monkeypatch)
    det = yolo_detector.YOLOFaceDetector()
    assert det.detect_faces(None) == []


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_empty_frame_returns_empty(monkeypatch):
    _patch_yolo(monkeypatch)
    det = yolo_detector.YOLOFaceDetector()
    assert det.detect_faces(np.zeros((0, 0, 3), dtype=np.uint8)) == []


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_no_boxes_returns_empty(monkeypatch, blank):
    _patch_yolo(monkeypatch, boxes_data=[])
    det = yolo_detector.YOLOFaceDetector()
    assert det.detect_faces(blank) == []


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_returns_correct_schema(monkeypatch, blank):
    _patch_yolo(monkeypatch, boxes_data=[([10.0, 20.0, 50.0, 60.0], 0.85)])
    det = yolo_detector.YOLOFaceDetector()
    faces = det.detect_faces(blank)
    assert len(faces) == 1
    f = faces[0]
    assert f["rect"] == (10, 20, 40, 40)
    assert f["center"] == (30, 40)
    assert f["confidence"] == pytest.approx(0.85)
    assert f["area"] == 40 * 40


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_skips_zero_width_box(monkeypatch, blank):
    # x1 == x2 → w = 0 → degenerate box, must be dropped
    _patch_yolo(monkeypatch, boxes_data=[([20.0, 10.0, 20.0, 50.0], 0.9)])
    det = yolo_detector.YOLOFaceDetector()
    assert det.detect_faces(blank) == []


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="ultralytics not installed")
def test_detect_faces_multiple_boxes(monkeypatch, blank):
    _patch_yolo(monkeypatch, boxes_data=[
        ([0.0, 0.0, 30.0, 30.0], 0.8),
        ([40.0, 40.0, 80.0, 80.0], 0.9),
    ])
    det = yolo_detector.YOLOFaceDetector()
    faces = det.detect_faces(blank)
    assert len(faces) == 2
