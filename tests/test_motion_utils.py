"""Unit tests for motion-detection frame validation and optical-flow path."""
import numpy as np

import motion_utils
from motion_utils import detect_motion, validate_frames


def valid_frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_validate_frames_accepts_matching_bgr_uint8_frames():
    assert validate_frames(valid_frame(), valid_frame()) is True


def test_validate_frames_rejects_none():
    assert validate_frames(None, valid_frame()) is False
    assert validate_frames(valid_frame(), None) is False


def test_validate_frames_rejects_shape_mismatch():
    a = valid_frame()
    b = np.zeros((24, 32, 3), dtype=np.uint8)
    assert validate_frames(a, b) is False


def test_validate_frames_rejects_non_uint8_dtype():
    a = np.zeros((48, 64, 3), dtype=np.float32)
    b = np.zeros((48, 64, 3), dtype=np.float32)
    assert validate_frames(a, b) is False


def test_validate_frames_rejects_single_channel():
    a = np.zeros((48, 64), dtype=np.uint8)
    b = np.zeros((48, 64), dtype=np.uint8)
    assert validate_frames(a, b) is False


def _moving_frames():
    prev = np.zeros((48, 64, 3), dtype=np.uint8)
    curr = np.zeros((48, 64, 3), dtype=np.uint8)
    prev[10:20, 10:20] = 255
    curr[12:22, 14:24] = 255  # shifted block -> optical flow motion
    return prev, curr


def test_detect_motion_returns_magnitude_array():
    prev, curr = _moving_frames()
    mag = detect_motion(prev, curr)
    assert isinstance(mag, np.ndarray)
    assert mag.shape == (48, 64)


def test_detect_motion_invalid_frames_returns_none():
    assert detect_motion(None, valid_frame()) is None


def test_detect_motion_debug_returns_tuple():
    prev, curr = _moving_frames()
    result = detect_motion(prev, curr, debug=True)
    assert isinstance(result, tuple)
    magnitude, vis = result
    assert isinstance(magnitude, np.ndarray)
    assert vis is not None and vis.shape == prev.shape


def test_detect_motion_handles_exception(monkeypatch):
    # Force the optical-flow call to raise -> outer except returns None.
    def boom(*a, **k):
        raise RuntimeError("flow exploded")

    monkeypatch.setattr(motion_utils.cv2, "calcOpticalFlowFarneback", boom)
    prev, curr = _moving_frames()
    assert detect_motion(prev, curr) is None


def test_detect_motion_cvtcolor_error_returns_none(monkeypatch):
    import cv2

    def boom(*a, **k):
        raise cv2.error("bad convert")

    monkeypatch.setattr(motion_utils.cv2, "cvtColor", boom)
    prev, curr = _moving_frames()
    assert detect_motion(prev, curr) is None
