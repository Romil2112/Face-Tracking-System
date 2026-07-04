"""Tests for the geometry IoU helper, including the malformed-input branch."""
import pytest

from geometry import calculate_iou


def test_iou_identical_boxes_is_one():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_partial_overlap():
    # Two 10x10 boxes overlapping in a 5x5 corner: inter=25, union=175.
    assert calculate_iou((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_iou_disjoint_boxes_is_zero():
    assert calculate_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_zero_area_union_returns_zero():
    # Degenerate zero-area boxes -> union_area == 0 -> 0.0 branch.
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_iou_malformed_input_returns_zero():
    # box1 cannot unpack into four values -> except branch returns 0.0.
    assert calculate_iou((0, 0), (0, 0, 0, 0)) == 0.0


def test_iou_non_numeric_input_returns_zero():
    assert calculate_iou(("a", "b", "c", "d"), (0, 0, 10, 10)) == 0.0
