"""Tests for VideoCapture using a fake cv2.VideoCapture (no real camera)."""
import numpy as np
import pytest

import video_capture
from video_capture import VideoCapture


class FakeCap:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, index=0, api=None, *, opened=True, frames=None):
        self.index = index
        self._opened = opened
        self._props = {}
        self.released = False
        # frames: list of (success, frame) tuples returned by retrieve().
        self._frames = frames if frames is not None else [
            (True, np.zeros((48, 64, 3), dtype=np.uint8))
        ]
        self._i = 0

    def isOpened(self):
        return self._opened and not self.released

    def grab(self):
        return True

    def retrieve(self):
        if self._i < len(self._frames):
            result = self._frames[self._i]
            self._i += 1
            return result
        return self._frames[-1]

    def set(self, prop, value):
        self._props[prop] = value
        return True

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 0

    def release(self):
        self.released = True
        self._opened = False


def _fast_capture():
    """A capture whose frame_time is 0 so read() never sleeps."""
    vc = VideoCapture(camera_index=0, width=640, height=480, fps=0)
    return vc


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(video_capture.time, "sleep", lambda *_: None)


@pytest.fixture(autouse=True)
def _dynamic_skip(monkeypatch):
    # _acquire only retrieves a frame when DYNAMIC_FRAME_SKIP is True.
    monkeypatch.setattr(video_capture.config, "DYNAMIC_FRAME_SKIP", True)


def test_start_success(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = _fast_capture()
    assert vc.start() is True
    assert vc.is_running is True
    vc.stop()


def test_start_failure_path(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, opened=False, **k))
    vc = _fast_capture()
    # A never-opening camera raises OSError inside start(); the except block runs
    # the error handler and returns False cleanly (no exception propagates).
    assert vc.start() is False
    assert vc.is_running is False


def test_read_success(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = _fast_capture()
    vc.start()
    ok, frame = vc.read()
    assert ok is True
    assert isinstance(frame, np.ndarray)
    vc.stop()


def test_read_not_running_returns_false():
    vc = _fast_capture()
    ok, frame = vc.read()
    assert ok is False
    assert frame is None


def test_read_empty_frame_triggers_recovery(monkeypatch):
    calls = {"n": 0}

    def make(*a, **k):
        # First capture yields a failed retrieve; recovery makes a fresh good one.
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeCap(*a, frames=[(False, None)], **k)
        return FakeCap(*a, **k)

    monkeypatch.setattr(video_capture.cv2, "VideoCapture", make)
    vc = _fast_capture()
    vc.start()
    ok, frame = vc.read()
    assert ok is False
    assert frame is None
    vc.stop()


def test_read_critical_error_recovers(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = _fast_capture()
    vc.start()

    # Make _acquire raise to hit read()'s outer except block, which runs the error
    # handler + recovery and returns (False, None) rather than propagating.
    def boom():
        raise RuntimeError("acquire failed")

    monkeypatch.setattr(vc, "_acquire", boom)
    ok, frame = vc.read()
    assert ok is False
    assert frame is None
    vc.stop()


def test_acquire_no_dynamic_skip(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    monkeypatch.setattr(video_capture.config, "DYNAMIC_FRAME_SKIP", False)
    vc = _fast_capture()
    vc.start()
    success, frame = vc._acquire()
    assert success is False
    assert frame is None
    vc.stop()


def test_acquire_paces_to_frame_time(monkeypatch):
    slept = {"val": None}
    monkeypatch.setattr(video_capture.time, "sleep",
                        lambda s: slept.__setitem__("val", s))
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = VideoCapture(camera_index=0, width=640, height=480, fps=30)
    vc.start()
    vc.last_frame_time = video_capture.time.time()  # force elapsed < frame_time
    vc._acquire()
    assert slept["val"] is not None  # sleep was called to pace
    vc.stop()


def test_recover_capture_handles_start_exception(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = _fast_capture()
    vc.start()

    def boom():
        raise RuntimeError("start failed during recovery")

    monkeypatch.setattr(vc, "start", boom)
    # Should swallow the exception from start().
    vc._recover_capture()


def test_stop_is_idempotent(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    vc = _fast_capture()
    vc.start()
    vc.stop()
    assert vc.cap is None
    assert vc.is_running is False
    vc.stop()  # second call, cap is None
    assert vc.is_running is False


def test_context_manager(monkeypatch):
    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FakeCap(*a, **k))
    with VideoCapture(camera_index=0, width=640, height=480, fps=0) as vc:
        assert vc.is_running is True
    assert vc.is_running is False


def test_set_camera_property_warns_on_failure(monkeypatch):
    class FailingSetCap(FakeCap):
        def set(self, prop, value):
            return False

    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: FailingSetCap(*a, **k))
    vc = _fast_capture()
    # start() calls _set_camera_property which logs a warning but doesn't raise.
    assert vc.start() is True
    vc.stop()


def test_start_resolution_mismatch_warns(monkeypatch):
    class MismatchCap(FakeCap):
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 320
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 240
            return 0

    monkeypatch.setattr(video_capture.cv2, "VideoCapture",
                        lambda *a, **k: MismatchCap(*a, **k))
    vc = _fast_capture()
    assert vc.start() is True  # mismatch only logs a warning
    vc.stop()
