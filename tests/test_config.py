"""Sanity checks on configuration invariants.

These guard against typos/regressions in the tunable parameters that the
detection and temporal-filtering pipelines depend on.
"""
import config


def test_confidence_thresholds_in_unit_range():
    for value in (
        config.DNN_CONFIDENCE_THRESHOLD,
        config.MINIMUM_CONFIDENCE,
        config.NMS_THRESHOLD,
    ):
        assert 0.0 <= value <= 1.0


def test_temporal_consistency_values_stay_in_sync():
    # config.py documents that these two must hold the same value.
    assert config.TEMPORAL_CONSISTENCY == config.TEMPORAL_CONSISTENCY_THRESHOLD


def test_frame_skip_bounds():
    assert config.FRAME_SKIP >= 1
    assert config.MAX_FRAME_SKIP >= config.FRAME_SKIP


def test_time_diff_bounds_are_ordered():
    assert 0 < config.TEMPORAL_MIN_TIME_DIFF < config.TEMPORAL_MAX_TIME_DIFF


def test_camera_dimensions_positive():
    assert config.CAMERA_WIDTH > 0
    assert config.CAMERA_HEIGHT > 0
    assert config.CAMERA_FPS > 0
