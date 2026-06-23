"""Unit tests for motion-detection frame validation."""
import numpy as np

from motion_utils import validate_frames


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
