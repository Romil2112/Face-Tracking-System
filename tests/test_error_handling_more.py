"""Tests for ErrorHandler recovery strategies and CUDA handling."""
import pytest

import config
import error_handling
from error_handling import (
    CameraRecoveryError,
    DnnDetectionError,
    ErrorHandler,
    HaarDetectionError,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(error_handling.time, "sleep", lambda *_: None)


@pytest.fixture
def handler():
    return ErrorHandler()


def test_filter_warnings_suppresses_opencl(handler):
    class Rec:
        def getMessage(self):
            return "error in ocl4dnn_conv_spatial.cpp build"

    assert handler.filter_warnings(Rec()) is False


def test_filter_warnings_passes_other_messages(handler):
    class Rec:
        def getMessage(self):
            return "some unrelated message"

    assert handler.filter_warnings(Rec()) is True


def test_handle_camera_error_recovers_via_first_strategy(handler):
    # _reconnect_camera returns True when the error carries a `.camera` with a
    # start() that returns True.
    class Cam:
        def stop(self):
            pass

        def start(self):
            return True

    err = RuntimeError("cam down")
    err.camera = Cam()
    assert handler.handle_camera_error(err) is True


def test_handle_camera_error_default_error(handler, monkeypatch):
    # No recovery strategy succeeds -> returns False and records a failure.
    monkeypatch.setattr(handler, "_reconnect_camera", lambda e: False)
    monkeypatch.setattr(handler, "_try_alternate_camera", lambda e: False)
    monkeypatch.setattr(handler, "_reset_camera_settings", lambda e: False)
    assert handler.handle_camera_error() is False


def test_handle_camera_error_breaker_open(handler, monkeypatch):
    monkeypatch.setattr(handler, "_reconnect_camera", lambda e: False)
    monkeypatch.setattr(handler, "_try_alternate_camera", lambda e: False)
    monkeypatch.setattr(handler, "_reset_camera_settings", lambda e: False)
    err = Exception("boom")
    for _ in range(3):
        handler.handle_camera_error(err)
    # Breaker for 'default' cam is now open.
    assert handler.handle_camera_error(err) is False


def test_reconnect_camera_no_camera_attr(handler):
    assert handler._reconnect_camera(Exception("x")) is False


def test_reconnect_camera_handles_exception(handler):
    class Cam:
        def stop(self):
            raise RuntimeError("stop failed")

    err = Exception("x")
    err.camera = Cam()
    assert handler._reconnect_camera(err) is False


def test_try_alternate_camera_finds_working(handler, monkeypatch):
    class Cap:
        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(error_handling.cv2, "VideoCapture", lambda idx: Cap())
    assert handler._try_alternate_camera(Exception("x")) is True


def test_try_alternate_camera_none_found(handler, monkeypatch):
    class Cap:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(error_handling.cv2, "VideoCapture", lambda idx: Cap())
    assert handler._try_alternate_camera(Exception("x")) is False


def test_try_alternate_camera_handles_cv2_error(handler, monkeypatch):
    def boom(idx):
        raise error_handling.cv2.error("no device")

    monkeypatch.setattr(error_handling.cv2, "VideoCapture", boom)
    assert handler._try_alternate_camera(Exception("x")) is False


def test_reset_camera_settings_with_camera(handler):
    class Cam:
        def stop(self):
            pass

        def start(self):
            return True

    err = Exception("x")
    err.camera = Cam()
    assert handler._reset_camera_settings(err) is True


def test_reset_camera_settings_no_camera(handler):
    assert handler._reset_camera_settings(Exception("x")) is False


def test_reset_camera_settings_handles_exception(handler):
    class Cam:
        def stop(self):
            raise RuntimeError("boom")

    err = Exception("x")
    err.camera = Cam()
    assert handler._reset_camera_settings(err) is False


def test_handle_face_detection_error_switch_dnn(handler):
    # _switch_detection_method returns True for a DnnDetectionError -> recovered.
    assert handler.handle_face_detection_error(DnnDetectionError("dnn")) is True


def test_switch_detection_method_haar(handler):
    assert handler._switch_detection_method(HaarDetectionError("haar")) is True


def test_switch_detection_method_other(handler):
    assert handler._switch_detection_method(Exception("other")) is False


def test_handle_face_detection_error_default(handler):
    # A plain error: _switch fails, _reload fails, _adjust returns True -> recovered.
    assert handler.handle_face_detection_error() is True


def test_handle_face_detection_error_breaker_open(handler, monkeypatch):
    monkeypatch.setattr(handler, "_switch_detection_method", lambda e: False)
    monkeypatch.setattr(handler, "_reload_detection_model", lambda e: False)
    monkeypatch.setattr(handler, "_adjust_detection_parameters", lambda e: False)
    err = Exception("boom")
    for _ in range(3):
        handler.handle_face_detection_error(err)
    assert handler.handle_face_detection_error(err) is False


def test_reload_detection_model_returns_false(handler):
    assert handler._reload_detection_model(Exception("x")) is False


def test_adjust_detection_parameters_returns_true(handler):
    assert handler._adjust_detection_parameters(Exception("x")) is True


def test_handle_resource_exhaustion_cuda_oom(handler, monkeypatch):
    monkeypatch.setattr(config, "MAX_FACES", 5)
    monkeypatch.setattr(config, "DYNAMIC_FRAME_SKIP", False)
    err = Exception("CUDA_OUT_OF_MEMORY occurred")
    # _handle_cuda_backend: 'CUDA' in message, and CUDA backend present in the
    # (real headless) build? If present it returns True immediately; otherwise it
    # falls through to the CUDA_OUT_OF_MEMORY branch. Either way -> True.
    assert handler.handle_resource_exhaustion(err) is True


def test_handle_resource_exhaustion_cuda_oom_branch(handler, monkeypatch):
    # Force _handle_cuda_backend to fall through (None) so the explicit
    # CUDA_OUT_OF_MEMORY branch (adjust MAX_FACES / DYNAMIC_FRAME_SKIP) runs.
    monkeypatch.setattr(handler, "_handle_cuda_backend", lambda e: None)
    monkeypatch.setattr(config, "MAX_FACES", 5)
    monkeypatch.setattr(config, "DYNAMIC_FRAME_SKIP", False)
    err = Exception("CUDA_OUT_OF_MEMORY hit")
    assert handler.handle_resource_exhaustion(err) is True
    assert config.DYNAMIC_FRAME_SKIP is True
    assert config.MAX_FACES == 4


def test_handle_resource_exhaustion_generic_recovery(handler, monkeypatch):
    # Non-CUDA error: _handle_cuda_backend returns None, breaker closed, and the
    # first recovery strategy (_reduce_processing_load) returns True.
    assert handler.handle_resource_exhaustion(Exception("disk full")) is True


def test_handle_resource_exhaustion_breaker_open(handler, monkeypatch):
    monkeypatch.setattr(handler, "_reduce_processing_load", lambda e: False)
    monkeypatch.setattr(handler, "_free_memory_resources", lambda e: False)
    monkeypatch.setattr(handler, "_restart_subsystems", lambda e: False)
    err = Exception("generic")
    for _ in range(3):
        handler.handle_resource_exhaustion(err)
    assert handler.handle_resource_exhaustion(err) is False


def test_handle_cuda_backend_removes_priority(handler, monkeypatch):
    # Force getAvailableBackends to omit CUDA so the "not available" branch runs.
    monkeypatch.setattr(error_handling.cv2.dnn, "getAvailableBackends",
                        lambda: [], raising=False)
    monkeypatch.setattr(config, "ACCELERATION_PRIORITY", ["CUDA", "OpenCL", "CPU"])
    result = handler._handle_cuda_backend(Exception("CUDA failure"))
    assert result is True
    assert "CUDA" not in config.ACCELERATION_PRIORITY


def test_handle_cuda_backend_priority_already_absent(handler, monkeypatch):
    # CUDA backend unavailable and "CUDA" already not in priority -> skip the
    # remove() but still return True (the 140->142 / 143 branch).
    monkeypatch.setattr(error_handling.cv2.dnn, "getAvailableBackends",
                        lambda: [], raising=False)
    monkeypatch.setattr(config, "ACCELERATION_PRIORITY", ["OpenCL", "CPU"])
    assert handler._handle_cuda_backend(Exception("CUDA failure")) is True


def test_handle_cuda_backend_non_cuda_returns_none(handler):
    assert handler._handle_cuda_backend(Exception("no gpu keyword")) is None


def test_reduce_processing_load_toggles_config(handler, monkeypatch):
    monkeypatch.setattr(config, "FRAME_SKIP", 1)
    monkeypatch.setattr(config, "TEMPORAL_FILTERING_ENABLED", True)
    monkeypatch.setattr(config, "MOTION_DETECTION_ENABLED", True)
    assert handler._reduce_processing_load(Exception("x")) is True
    assert config.TEMPORAL_FILTERING_ENABLED is False
    assert config.MOTION_DETECTION_ENABLED is False
    assert config.FRAME_SKIP == 2


def test_free_memory_resources(handler):
    assert handler._free_memory_resources(Exception("x")) is True


def test_restart_subsystems(handler):
    assert handler._restart_subsystems(Exception("x")) is True


def test_custom_exceptions_are_exceptions():
    assert issubclass(DnnDetectionError, Exception)
    assert issubclass(HaarDetectionError, Exception)
    assert issubclass(CameraRecoveryError, Exception)
