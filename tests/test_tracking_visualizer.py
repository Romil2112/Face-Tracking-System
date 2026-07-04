"""Tests for TrackingVisualizer drawing helpers (headless, numpy frames only)."""
import numpy as np
import pytest

from tracking_visualizer import TrackingVisualizer


@pytest.fixture
def viz():
    return TrackingVisualizer()


@pytest.fixture
def frame():
    return np.zeros((120, 120, 3), dtype=np.uint8)


def test_init_clamps_minimum_values():
    v = TrackingVisualizer(rect_thickness=0, center_radius=0,
                           font_scale=0.1, font_thickness=0)
    assert v.rect_thickness == 1
    assert v.center_radius == 1
    assert v.font_scale == 0.5
    assert v.font_thickness == 1


def test_draw_faces_returns_same_shape(viz, frame):
    faces = [{"rect": (10, 10, 30, 30), "center": (25, 25), "confidence": 0.9}]
    out = viz.draw_faces(frame, faces, 30.0)
    assert out.shape == frame.shape
    # Should have drawn something (non-zero pixels).
    assert out.sum() > 0


def test_draw_faces_none_frame_returns_placeholder(viz):
    out = viz.draw_faces(None, [], 30.0)
    assert out.shape == (100, 100, 3)


def test_draw_faces_empty_frame_returns_placeholder(viz):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    out = viz.draw_faces(empty, [], 30.0)
    assert out.shape == (100, 100, 3)


def test_draw_faces_no_faces(viz, frame):
    out = viz.draw_faces(frame, [], 12.5)
    # FPS text is still drawn.
    assert out.shape == frame.shape


def test_draw_one_face_skips_nan_rect(viz, frame):
    before = frame.copy()
    faces = [{"rect": (np.nan, np.nan, np.nan, np.nan),
              "center": (10, 10), "confidence": 0.9}]
    out = viz.draw_faces(frame, faces, 30.0)
    # NaN rect -> _draw_one_face returns early; only FPS text drawn, not a box.
    assert out.shape == before.shape


def test_draw_one_face_missing_keys_handled(viz, frame):
    # A face without 'rect' triggers the KeyError/TypeError guard.
    faces = [{"confidence": 0.5}]
    out = viz.draw_faces(frame, faces, 30.0)
    assert out.shape == frame.shape


def test_draw_one_face_without_center_or_confidence(viz, frame):
    faces = [{"rect": (5, 5, 20, 20)}]
    out = viz.draw_faces(frame, faces, 30.0)
    assert out.shape == frame.shape


def test_safe_draw_rect_out_of_bounds(viz, frame):
    # Coordinates beyond the frame get clamped; should not raise.
    viz._safe_draw_rect(frame, -50, -50, 500, 500)


def test_safe_draw_rect_degenerate_no_draw(viz, frame):
    # x2 <= x1 means nothing is drawn (the `if x2 > x1 and y2 > y1` guard).
    before = frame.copy()
    viz._safe_draw_rect(frame, 10, 10, 0, 0)
    assert np.array_equal(frame, before)


def test_safe_draw_rect_bad_frame_logs(viz):
    # Passing a non-array triggers the except branch (no crash).
    viz._safe_draw_rect(object(), 0, 0, 10, 10)


def test_safe_draw_center_out_of_bounds(viz, frame):
    viz._safe_draw_center(frame, (1000, 1000))


def test_safe_draw_center_bad_frame_logs(viz):
    viz._safe_draw_center(object(), (5, 5))
