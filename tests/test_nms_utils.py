"""Unit tests for Non-Maximum Suppression helpers."""
import numpy as np
import pytest

import nms_utils
from geometry import calculate_iou
from nms_utils import _manual_iou_filter, _prepare_boxes, apply_nms


def make_face(rect, confidence=0.9):
    return {"rect": rect, "confidence": confidence}


def test_iou_identical_boxes_is_one():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


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


def test_prepare_boxes_skips_unpackable_rect():
    # First entry has a valid rect; second's rect cannot unpack into 4 -> skipped.
    faces = [
        {"rect": (0, 0, 10, 10), "confidence": 0.9},
        {"rect": (1, 2), "confidence": 0.5},
    ]
    boxes, scores, valid = _prepare_boxes(faces)
    assert len(boxes) == 1
    assert len(valid) == 1
    assert boxes[0] == [0, 0, 10, 10]


def test_prepare_boxes_default_confidence():
    boxes, scores, valid = _prepare_boxes([{"rect": (0, 0, 10, 10)}])
    assert scores == [0.5]


def test_prepare_boxes_rejects_batch_missing_rect():
    assert _prepare_boxes([{"confidence": 0.9}]) == ([], [], [])


def test_manual_iou_filter_suppresses_overlap():
    faces = [
        make_face((0, 0, 100, 100), confidence=0.9),
        make_face((5, 5, 100, 100), confidence=0.4),
    ]
    kept = _manual_iou_filter(faces, threshold=0.4)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_manual_iou_filter_keeps_distinct():
    faces = [
        make_face((0, 0, 20, 20), confidence=0.9),
        make_face((500, 500, 20, 20), confidence=0.8),
    ]
    assert len(_manual_iou_filter(faces, threshold=0.4)) == 2


def test_apply_nms_falls_back_to_manual_on_cv2_error(monkeypatch):
    def boom(*a, **k):
        raise nms_utils.cv2.error("NMSBoxes unavailable")

    monkeypatch.setattr(nms_utils.cv2.dnn, "NMSBoxes", boom)
    faces = [
        make_face((0, 0, 100, 100), confidence=0.9),
        make_face((5, 5, 100, 100), confidence=0.4),
    ]
    kept = apply_nms(faces, overlap_threshold=0.4)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.9


def test_apply_nms_none_indices_returns_empty(monkeypatch):
    monkeypatch.setattr(nms_utils.cv2.dnn, "NMSBoxes", lambda *a, **k: None)
    faces = [
        make_face((0, 0, 50, 50), confidence=0.9),
        make_face((400, 400, 50, 50), confidence=0.8),
    ]
    assert apply_nms(faces) == []


def test_apply_nms_handles_list_indices(monkeypatch):
    # Some OpenCV versions return a plain list rather than an ndarray.
    monkeypatch.setattr(nms_utils.cv2.dnn, "NMSBoxes", lambda *a, **k: [0])
    faces = [
        make_face((0, 0, 50, 50), confidence=0.9),
        make_face((400, 400, 50, 50), confidence=0.8),
    ]
    kept = apply_nms(faces)
    assert len(kept) == 1


def test_apply_nms_flattens_ndarray_indices(monkeypatch):
    monkeypatch.setattr(nms_utils.cv2.dnn, "NMSBoxes",
                        lambda *a, **k: np.array([[0], [1]]))
    faces = [
        make_face((0, 0, 50, 50), confidence=0.9),
        make_face((400, 400, 50, 50), confidence=0.8),
    ]
    assert len(apply_nms(faces)) == 2
