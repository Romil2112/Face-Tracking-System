"""Tests for the main application module (headless, mocked GUI + camera)."""
import numpy as np
import pytest

import config
import main
from face_detector import FaceDetector


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


@pytest.fixture
def blank():
    return np.zeros((120, 120, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)


# --------------------------- parse_arguments ---------------------------------

def test_parse_arguments_defaults(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["prog"])
    args = main.parse_arguments()
    assert args["camera"] == config.CAMERA_INDEX
    assert args["max_faces"] == config.MAX_FACES
    assert args["debug"] is False


def test_parse_arguments_overrides(monkeypatch):
    monkeypatch.setattr(main.sys, "argv",
                        ["prog", "--camera", "2", "--max-faces", "3", "--debug"])
    args = main.parse_arguments()
    assert args["camera"] == 2
    assert args["max_faces"] == 3
    assert args["debug"] is True


# --------------------------- verify_acceleration -----------------------------

def test_verify_acceleration_returns_bool():
    assert isinstance(main.verify_acceleration(), bool)


def test_verify_acceleration_handles_exception(monkeypatch):
    monkeypatch.setattr(main.acceleration, "probe_capabilities",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert main.verify_acceleration() is False


def test_verify_acceleration_gpu_true(monkeypatch):
    caps = {"cuda": True, "opencl": True}
    monkeypatch.setattr(main.acceleration, "probe_capabilities", lambda: caps)
    accel = main.acceleration.Acceleration(main.acceleration.CUDA, 0, 0)
    monkeypatch.setattr(main.acceleration, "select_acceleration",
                        lambda caps=None: accel)
    assert main.verify_acceleration() is True


# --------------------------- validate_configuration --------------------------

def test_validate_configuration_ok():
    # Real model + cascade files exist in the repo.
    main.validate_configuration()


def test_validate_configuration_missing_file(monkeypatch):
    monkeypatch.setattr(main.config, "CASCADE_PATH", "/no/such/cascade.xml")
    with pytest.raises(FileNotFoundError):
        main.validate_configuration()


def test_validate_configuration_bad_threshold(monkeypatch):
    monkeypatch.setattr(main.config, "DNN_CONFIDENCE_THRESHOLD", 1.5)
    with pytest.raises(ValueError, match="confidence threshold"):
        main.validate_configuration()


def test_validate_configuration_negative_camera(monkeypatch):
    monkeypatch.setattr(main.config, "CAMERA_INDEX", -1)
    with pytest.raises(ValueError, match="Camera index"):
        main.validate_configuration()


def test_validate_cuda_backend_no_attr(monkeypatch):
    # getAvailableBackends missing -> AttributeError -> defaults used. As long as
    # CUDA targets resolve, no error is raised (headless build has no CUDA).
    main._validate_cuda_backend()


# --------------------------- detection helpers -------------------------------

def test_detect_and_filter(detector, blank):
    faces = main._detect_and_filter(blank, detector, {"max_faces": 5})
    assert isinstance(faces, list)


def test_apply_motion_gate_no_motion_returns_all(monkeypatch, blank):
    faces = [{"rect": (0, 0, 20, 20), "confidence": 0.9}]
    # detect_motion returning None -> faces passed through unchanged.
    monkeypatch.setitem(__import__("sys").modules, "motion_utils",
                        _FakeMotion(None))
    result = main._apply_motion_gate(faces, blank, blank)
    assert result == faces


def test_apply_motion_gate_filters_by_threshold(monkeypatch, blank):
    faces = [{"rect": (0, 0, 20, 20), "confidence": 0.9}]
    mag = np.full((120, 120), 100.0)  # high motion everywhere
    monkeypatch.setitem(__import__("sys").modules, "motion_utils",
                        _FakeMotion(mag))
    monkeypatch.setattr(main.config, "MOTION_THRESHOLD", 5.0)
    result = main._apply_motion_gate(faces, blank, blank)
    assert result == faces


def test_apply_motion_gate_drops_low_motion(monkeypatch, blank):
    faces = [{"rect": (0, 0, 20, 20), "confidence": 0.9}]
    mag = np.zeros((120, 120))  # no motion
    monkeypatch.setitem(__import__("sys").modules, "motion_utils",
                        _FakeMotion(mag))
    monkeypatch.setattr(main.config, "MOTION_THRESHOLD", 5.0)
    result = main._apply_motion_gate(faces, blank, blank)
    assert result == []


def test_apply_motion_gate_handles_exception(monkeypatch, blank):
    faces = [{"rect": (0, 0, 20, 20), "confidence": 0.9}]
    monkeypatch.setitem(__import__("sys").modules, "motion_utils",
                        _FakeMotion(RuntimeError("boom")))
    # Exception inside -> returns the original faces.
    assert main._apply_motion_gate(faces, blank, blank) == faces


class _FakeMotion:
    """Fake motion_utils module whose detect_motion returns a fixed value or raises."""

    def __init__(self, value):
        self._value = value

    def detect_motion(self, prev, curr):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakeTemporal:
    def __init__(self):
        self.reset_called = False

    def update(self, faces, time_diff):
        return faces

    def reset(self):
        self.reset_called = True


def test_apply_temporal_passthrough(blank):
    faces = [{"rect": (0, 0, 10, 10), "confidence": 0.9}]
    tf = _FakeTemporal()
    assert main._apply_temporal(faces, tf, blank) == faces


def test_apply_temporal_none_prev_frame():
    faces = [{"rect": (0, 0, 10, 10), "confidence": 0.9}]
    tf = _FakeTemporal()
    assert main._apply_temporal(faces, tf, None) == faces


def test_apply_temporal_handles_exception(blank):
    faces = [{"rect": (0, 0, 10, 10), "confidence": 0.9}]

    class BadTemporal:
        def update(self, faces, time_diff):
            raise RuntimeError("temporal boom")

    assert main._apply_temporal(faces, BadTemporal(), blank) == faces


# --------------------------- process_frame -----------------------------------

def test_process_frame_basic(detector, blank):
    faces = main.process_frame(blank, detector, {"max_faces": 5},
                               motion_detection=False, prev_frame=None,
                               temporal_filter=None)
    assert isinstance(faces, list)


def test_process_frame_with_motion_and_temporal(detector, blank, monkeypatch):
    monkeypatch.setattr(main.config, "MOTION_DETECTION_ENABLED", True)
    monkeypatch.setitem(__import__("sys").modules, "motion_utils",
                        _FakeMotion(None))
    faces = main.process_frame(blank, detector, {"max_faces": 5},
                               motion_detection=True, prev_frame=blank,
                               temporal_filter=_FakeTemporal())
    assert isinstance(faces, list)


def test_process_frame_handles_exception(detector, blank, monkeypatch):
    monkeypatch.setattr(main, "_detect_and_filter",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert main.process_frame(blank, detector, {"max_faces": 5},
                              False, None, None) == []


# --------------------------- _next_frame -------------------------------------

def test_next_frame_success():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    class VC:
        def read(self):
            return True, frame

    result, retry = main._next_frame(VC(), retry_count=2, max_retries=3)
    assert result is frame
    assert retry == 0


def test_next_frame_bad_read_increments():
    class VC:
        def read(self):
            return False, None

    result, retry = main._next_frame(VC(), retry_count=0, max_retries=3)
    assert result is None
    assert retry == 1


def test_next_frame_exhausts_retries_raises():
    class VC:
        def read(self):
            return False, None

    with pytest.raises(OSError, match="Maximum retry"):
        main._next_frame(VC(), retry_count=2, max_retries=3)


# --------------------------- _detect_with_reset ------------------------------

def test_detect_with_reset_normal(detector, blank, monkeypatch):
    monkeypatch.setattr(main, "process_frame", lambda *a, **k: ["face"])
    result = main._detect_with_reset(blank, detector, {"max_faces": 5},
                                     False, None, _FakeTemporal(), [])
    assert result == ["face"]


def test_detect_with_reset_recovers_datetime_typeerror(detector, blank, monkeypatch):
    msg = "unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'"

    def boom(*a, **k):
        raise TypeError(msg)

    monkeypatch.setattr(main, "process_frame", boom)
    tf = _FakeTemporal()
    prev_faces = ["old"]
    result = main._detect_with_reset(blank, detector, {"max_faces": 5},
                                     False, None, tf, prev_faces)
    assert result == prev_faces
    assert tf.reset_called is True


def test_detect_with_reset_reraises_other_typeerror(detector, blank, monkeypatch):
    def boom(*a, **k):
        raise TypeError("some other type error")

    monkeypatch.setattr(main, "process_frame", boom)
    with pytest.raises(TypeError):
        main._detect_with_reset(blank, detector, {"max_faces": 5},
                                False, None, _FakeTemporal(), [])


# --------------------------- _render / _should_quit --------------------------

class _FakeFPS:
    def update(self):
        pass

    def stop(self):
        pass

    def fps(self):
        return 30.0


def test_render_calls_imshow(monkeypatch, blank):
    shown = {}
    monkeypatch.setattr(main.cv2, "imshow",
                        lambda name, frame: shown.update(name=name, frame=frame))
    viz = main.TrackingVisualizer()
    main._render(viz, blank, [], _FakeFPS())
    assert shown["name"] == "Face Tracking"


def test_should_quit_true_on_q(monkeypatch):
    monkeypatch.setattr(main.cv2, "waitKey", lambda n: ord('q'))
    monkeypatch.setattr(main.cv2, "getWindowProperty", lambda *a: 1)
    assert main._should_quit() is True


def test_should_quit_true_on_closed_window(monkeypatch):
    monkeypatch.setattr(main.cv2, "waitKey", lambda n: 0)
    monkeypatch.setattr(main.cv2, "getWindowProperty", lambda *a: 0)
    assert main._should_quit() is True


def test_should_quit_false(monkeypatch):
    monkeypatch.setattr(main.cv2, "waitKey", lambda n: 0)
    monkeypatch.setattr(main.cv2, "getWindowProperty", lambda *a: 1)
    assert main._should_quit() is False


# --------------------------- _handle_loop_error ------------------------------

def test_handle_loop_error_recovered():
    class Handler:
        def handle_camera_error(self, e):
            return True

    # Recovered -> does not raise.
    main._handle_loop_error(RuntimeError("x"), Handler())


def test_handle_loop_error_reraises():
    class Handler:
        def handle_camera_error(self, e):
            return False

    with pytest.raises(RuntimeError):
        main._handle_loop_error(RuntimeError("x"), Handler())


# --------------------------- main_loop ---------------------------------------

def _patch_gui(monkeypatch, waitkey_seq):
    seq = iter(waitkey_seq)
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda n: next(seq, ord('q')))
    monkeypatch.setattr(main.cv2, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(main.cv2, "flip", lambda f, code: f)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)


class _FakeVC:
    def __init__(self, frame):
        self._frame = frame
        self.stopped = False

    def read(self):
        return True, self._frame

    def stop(self):
        self.stopped = True


def test_main_loop_runs_and_quits(monkeypatch, detector, blank):
    _patch_gui(monkeypatch, [0, ord('q')])
    monkeypatch.setattr(main, "process_frame", lambda *a, **k: [])
    vc = _FakeVC(blank)
    viz = main.TrackingVisualizer()

    class Handler:
        def handle_camera_error(self, e):
            return True

    main.main_loop(vc, detector, viz, {"max_faces": 5},
                   motion_detection=False, temporal_filter=None,
                   error_handler=Handler())
    assert vc.stopped is True


def test_main_loop_with_motion_detection(monkeypatch, detector, blank):
    _patch_gui(monkeypatch, [ord('q')])
    monkeypatch.setattr(main, "_detect_with_reset", lambda *a, **k: [])
    vc = _FakeVC(blank)
    viz = main.TrackingVisualizer()

    class Handler:
        def handle_camera_error(self, e):
            return True

    main.main_loop(vc, detector, viz, {"max_faces": 5},
                   motion_detection=True, temporal_filter=None,
                   error_handler=Handler())
    assert vc.stopped is True


def test_main_loop_skips_none_frame(monkeypatch, detector, blank):
    # First read fails (None frame -> continue), second succeeds then quit.
    reads = iter([(False, None), (True, blank)])

    class VC:
        def __init__(self):
            self.stopped = False

        def read(self):
            return next(reads, (True, blank))

        def stop(self):
            self.stopped = True

    _patch_gui(monkeypatch, [ord('q')])
    monkeypatch.setattr(main, "process_frame", lambda *a, **k: [])
    vc = VC()
    viz = main.TrackingVisualizer()

    class Handler:
        def handle_camera_error(self, e):
            return True

    main.main_loop(vc, detector, viz, {"max_faces": 5},
                   motion_detection=False, temporal_filter=None,
                   error_handler=Handler())
    assert vc.stopped is True


def test_main_loop_handles_error(monkeypatch, detector, blank):
    _patch_gui(monkeypatch, [ord('q')])

    def boom(*a, **k):
        raise RuntimeError("render boom")

    monkeypatch.setattr(main, "_render", boom)
    vc = _FakeVC(blank)
    viz = main.TrackingVisualizer()

    class Handler:
        def __init__(self):
            self.handled = False

        def handle_camera_error(self, e):
            self.handled = True
            return True  # recovered -> no re-raise

    h = Handler()
    main.main_loop(vc, detector, viz, {"max_faces": 5},
                   motion_detection=False, temporal_filter=None,
                   error_handler=h)
    assert h.handled is True
    assert vc.stopped is True


# --------------------------- initialize_video_capture ------------------------

def test_initialize_video_capture_success(monkeypatch):
    class VC:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return True

    monkeypatch.setattr(main, "VideoCapture", VC)
    result = main.initialize_video_capture(0, 640, 480, 30)
    assert isinstance(result, VC)


def test_initialize_video_capture_failure_raises(monkeypatch):
    class VC:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return False

    monkeypatch.setattr(main, "VideoCapture", VC)
    # start() False -> OSError (an IOError, in the retry allow-list) -> retried
    # and eventually re-raised.
    with pytest.raises(OSError):
        main.initialize_video_capture(0, 640, 480, 30)


def test_validate_cuda_backend_target_missing(monkeypatch):
    dnn = main.cv2.dnn
    monkeypatch.setattr(dnn, "getAvailableBackends",
                        lambda: [dnn.DNN_BACKEND_CUDA], raising=False)
    monkeypatch.setattr(dnn, "getAvailableTargets",
                        lambda backend: [], raising=False)
    with pytest.raises(RuntimeError, match="CUDA backend available"):
        main._validate_cuda_backend()


# --------------------------- main() ------------------------------------------

def test_main_success(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["prog"])
    monkeypatch.setattr(main, "validate_configuration", lambda: None)

    fake_detector = object()
    monkeypatch.setattr(main, "FaceDetector", lambda **k: fake_detector)

    class VC:
        def start(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(main, "initialize_video_capture", lambda *a, **k: VC())
    monkeypatch.setattr(main, "verify_acceleration", lambda: False)
    monkeypatch.setattr(main, "TrackingVisualizer", lambda **k: object())
    called = {"loop": False}
    monkeypatch.setattr(main, "main_loop",
                        lambda *a, **k: called.__setitem__("loop", True))
    monkeypatch.setattr(main.config, "TEMPORAL_FILTERING_ENABLED", False)
    monkeypatch.setattr(main.config, "MOTION_DETECTION_ENABLED", False)

    main.main()
    assert called["loop"] is True


def test_main_critical_error_exits(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["prog"])
    monkeypatch.setattr(main, "validate_configuration",
                        lambda: (_ for _ in ()).throw(RuntimeError("bad config")))
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1


def test_main_debug_and_temporal(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["prog", "--debug"])
    monkeypatch.setattr(main, "validate_configuration", lambda: None)
    monkeypatch.setattr(main, "FaceDetector", lambda **k: object())

    class VC:
        def start(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(main, "initialize_video_capture", lambda *a, **k: VC())
    monkeypatch.setattr(main, "verify_acceleration", lambda: True)
    monkeypatch.setattr(main, "TrackingVisualizer", lambda **k: object())
    monkeypatch.setattr(main, "main_loop", lambda *a, **k: None)
    monkeypatch.setattr(main.config, "TEMPORAL_FILTERING_ENABLED", True)
    monkeypatch.setattr(main.config, "MOTION_DETECTION_ENABLED", True)
    # Should construct a real TemporalFilter and probe motion_utils, then run.
    main.main()


def test_main_temporal_and_motion_failures(monkeypatch):
    import sys as _sys

    monkeypatch.setattr(main.sys, "argv", ["prog"])
    monkeypatch.setattr(main, "validate_configuration", lambda: None)
    monkeypatch.setattr(main, "FaceDetector", lambda **k: object())

    class VC:
        def start(self):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(main, "initialize_video_capture", lambda *a, **k: VC())
    monkeypatch.setattr(main, "verify_acceleration", lambda: False)
    monkeypatch.setattr(main, "TrackingVisualizer", lambda **k: object())
    monkeypatch.setattr(main, "main_loop", lambda *a, **k: None)
    monkeypatch.setattr(main.config, "TEMPORAL_FILTERING_ENABLED", True)
    monkeypatch.setattr(main.config, "MOTION_DETECTION_ENABLED", True)

    # Make `from temporal_filter import TemporalFilter` raise -> warning branch.
    class BadTemporalModule:
        def __getattr__(self, name):
            raise ImportError("temporal broken")

    monkeypatch.setitem(_sys.modules, "temporal_filter", BadTemporalModule())
    # Make `from motion_utils import detect_motion` raise ImportError -> disabled.
    monkeypatch.setitem(_sys.modules, "motion_utils", None)

    main.main()  # both failure branches exercised, no exception
