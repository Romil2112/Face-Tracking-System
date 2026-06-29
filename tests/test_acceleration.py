"""Unit tests for hardware-acceleration selection.

The selection logic is exercised with synthetic capability dicts so these run on
a CI runner that has neither a CUDA device nor a GPU OpenCL driver.
"""
import cv2
import numpy as np

import acceleration


CUDA_TARGET = getattr(cv2.dnn, "DNN_TARGET_CUDA", 6)


def test_prefers_cuda_when_available_and_first():
    caps = {"cuda": True, "cuda_targets": [CUDA_TARGET], "opencl": True}
    accel = acceleration.select_acceleration(["CUDA", "OpenCL", "CPU"], caps)
    assert accel.name == acceleration.CUDA


def test_falls_back_to_opencl_when_no_cuda():
    caps = {"cuda": False, "cuda_targets": [], "opencl": True}
    accel = acceleration.select_acceleration(["CUDA", "OpenCL", "CPU"], caps)
    assert accel.name == acceleration.OPENCL


def test_falls_back_to_cpu_when_nothing_available():
    caps = {"cuda": False, "cuda_targets": [], "opencl": False}
    accel = acceleration.select_acceleration(["CUDA", "OpenCL", "CPU"], caps)
    assert accel.name == acceleration.CPU


def test_priority_order_is_respected():
    # CUDA present but not in the priority list -> OpenCL wins.
    caps = {"cuda": True, "cuda_targets": [CUDA_TARGET], "opencl": True}
    accel = acceleration.select_acceleration(["OpenCL", "CPU"], caps)
    assert accel.name == acceleration.OPENCL


def test_cuda_skipped_when_target_missing():
    # CUDA device counts but the CUDA dnn target isn't built in -> OpenCL.
    caps = {"cuda": True, "cuda_targets": [], "opencl": True}
    accel = acceleration.select_acceleration(["CUDA", "OpenCL", "CPU"], caps)
    assert accel.name == acceleration.OPENCL


def test_empty_priority_defaults_to_cpu():
    caps = {"cuda": True, "cuda_targets": [CUDA_TARGET], "opencl": True}
    accel = acceleration.select_acceleration([], caps)
    assert accel.name == acceleration.CPU


def test_apply_to_net_sets_backend_and_target():
    class FakeNet:
        def setPreferableBackend(self, b): self.backend = b
        def setPreferableTarget(self, t): self.target = t

    net = FakeNet()
    accel = acceleration.Acceleration(
        acceleration.CPU, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    )
    returned = acceleration.apply_to_net(net, accel)
    assert returned is accel
    assert net.backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert net.target == cv2.dnn.DNN_TARGET_CPU


def test_to_umat_is_noop_for_cpu():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    accel = acceleration.Acceleration(
        acceleration.CPU, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    )
    assert acceleration.to_umat(frame, accel) is frame


def test_to_umat_falls_back_and_warns_when_umat_fails(monkeypatch, caplog):
    # Force the OpenCL T-API path, then make UMat construction blow up. The
    # frame must pass through unchanged (CPU fallback) and a warning logged.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    accel = acceleration.Acceleration(
        acceleration.OPENCL,
        cv2.dnn.DNN_BACKEND_OPENCV,
        getattr(cv2.dnn, "DNN_TARGET_OPENCL", 0),
    )
    monkeypatch.setattr(cv2.ocl, "useOpenCL", lambda: True)

    def boom(_frame):
        raise cv2.error("UMat unavailable")

    monkeypatch.setattr(cv2, "UMat", boom)
    with caplog.at_level("WARNING"):
        assert acceleration.to_umat(frame, accel) is frame
    assert "falling back to CPU" in caplog.text


def test_probe_capabilities_returns_expected_keys():
    caps = acceleration.probe_capabilities()
    assert set(caps) == {"cuda", "cuda_targets", "opencl"}
    assert isinstance(caps["opencl"], bool)
