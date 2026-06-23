"""Unit tests for TemporalFilter (pure logic - no camera or GPU required)."""
from datetime import timedelta

import pytest

from temporal_filter import TemporalFilter


def make_face(rect, confidence=0.9):
    x, y, w, h = rect
    return {"rect": rect, "center": (x + w // 2, y + h // 2), "confidence": confidence}


@pytest.fixture
def tf():
    return TemporalFilter(history_size=5, consistency_threshold=2)


def test_iou_identical_boxes_is_one(tf):
    assert tf._calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero(tf):
    assert tf._calculate_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap(tf):
    # Boxes overlap on half their width -> intersection 50 / union 150.
    assert tf._calculate_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_iou_handles_degenerate_union_gracefully(tf):
    # Zero-area boxes must not raise (division-by-zero guard).
    assert tf._calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_validate_face_accepts_well_formed_face(tf):
    assert tf._validate_face(make_face((0, 0, 20, 20))) is True


def test_validate_face_rejects_missing_keys(tf):
    assert tf._validate_face({"rect": (0, 0, 10, 10)}) is False


def test_validate_face_rejects_nonpositive_dimensions(tf):
    assert tf._validate_face(make_face((0, 0, 0, 10))) is False


def test_update_rejects_non_list_input(tf):
    assert tf.update("not-a-list", timedelta(milliseconds=33)) == []


def test_update_passes_face_meeting_consistency_threshold(tf):
    # consistency_threshold=2 is satisfied by a confirmed sighting.
    out = tf.update([make_face((0, 0, 20, 20))], timedelta(milliseconds=33))
    assert len(out) == 1
    assert out[0]["consistency"] >= 2


def test_update_filters_face_below_consistency_threshold():
    strict = TemporalFilter(history_size=5, consistency_threshold=3)
    assert strict.update([make_face((0, 0, 20, 20))], timedelta(milliseconds=33)) == []


def test_reset_clears_history(tf):
    tf.update([make_face((0, 0, 20, 20))], timedelta(milliseconds=33))
    assert len(tf.face_history) > 0
    tf.reset()
    assert len(tf.face_history) == 0
