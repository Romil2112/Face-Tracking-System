"""Unit tests for Non-Maximum Suppression helpers."""
import pytest

from nms_utils import _calculate_iou, apply_nms


def make_face(rect, confidence=0.9):
    return {"rect": rect, "confidence": confidence}


def test_iou_identical_boxes_is_one():
    assert _calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert _calculate_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_apply_nms_empty_input_returns_empty():
    assert apply_nms([]) == []


def test_apply_nms_rejects_entries_without_rect():
    assert apply_nms([{"confidence": 0.9}]) == []


def test_apply_nms_single_face_passes_through():
    faces = [make_face((0, 0, 100, 100))]
    assert len(apply_nms(faces)) == 1


def test_apply_nms_suppresses_heavily_overlapping_box():
    faces = [
        make_face((0, 0, 100, 100), confidence=0.9),
        make_face((10, 10, 100, 100), confidence=0.5),
    ]
    kept = apply_nms(faces)
    assert len(kept) == 1
    # The higher-confidence detection should be the survivor.
    assert kept[0]["confidence"] == 0.9


def test_apply_nms_keeps_distinct_non_overlapping_boxes():
    faces = [
        make_face((0, 0, 50, 50), confidence=0.9),
        make_face((400, 400, 50, 50), confidence=0.8),
    ]
    assert len(apply_nms(faces)) == 2
