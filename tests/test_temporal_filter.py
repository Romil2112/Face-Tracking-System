"""Unit tests for TemporalFilter (pure logic - no camera or GPU required)."""
from datetime import timedelta

import pytest

from geometry import calculate_iou
from temporal_filter import TemporalFilter


def make_face(rect, confidence=0.9):
    x, y, w, h = rect
    return {"rect": rect, "center": (x + w // 2, y + h // 2), "confidence": confidence}


@pytest.fixture
def tf():
    return TemporalFilter(history_size=5, consistency_threshold=2)


def test_iou_identical_boxes_is_one():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap():
    # Boxes overlap on half their width -> intersection 50 / union 150.
    assert calculate_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_iou_handles_degenerate_union_gracefully():
    # Zero-area boxes must not raise (division-by-zero guard).
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_validate_face_accepts_well_formed_face(tf):
    assert tf._validate_face(make_face((0, 0, 20, 20))) is True


def test_validate_face_rejects_missing_keys(tf):
    assert tf._validate_face({"rect": (0, 0, 10, 10)}) is False


def test_validate_face_rejects_nonpositive_dimensions(tf):
    assert tf._validate_face(make_face((0, 0, 0, 10))) is False


def test_update_rejects_non_list_input(tf):
    assert tf.update("not-a-list", timedelta(milliseconds=33)) == []


def test_update_first_sighting_not_yet_consistent(tf):
    # consistency_threshold=2: a single frame cannot be confirmed yet.
    out = tf.update([make_face((0, 0, 20, 20))], timedelta(milliseconds=33))
    assert out == []


def test_update_promotes_face_after_repeated_consistent_sightings(tf):
    rect = (0, 0, 20, 20)
    dt = timedelta(milliseconds=33)
    assert tf.update([make_face(rect)], dt) == []        # frame 1: consistency 1
    out = tf.update([make_face(rect)], dt)               # frame 2: consistency 2 -> passes
    assert len(out) == 1
    assert out[0]["consistency"] >= 2


def test_update_filters_face_below_consistency_threshold():
    strict = TemporalFilter(history_size=5, consistency_threshold=3)
    rect = (0, 0, 20, 20)
    dt = timedelta(milliseconds=33)
    # Even two consistent sightings only reach consistency 2, below the threshold of 3.
    strict.update([make_face(rect)], dt)
    assert strict.update([make_face(rect)], dt) == []


def test_reset_clears_history(tf):
    tf.update([make_face((0, 0, 20, 20))], timedelta(milliseconds=33))
    assert len(tf.face_history) > 0
    tf.reset()
    assert len(tf.face_history) == 0
