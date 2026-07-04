"""
Hardware Acceleration Selection
Author: Romil V. Shah

Chooses the detection backend by walking ``config.ACCELERATION_PRIORITY`` and
picking the first capability that is actually present:

    CUDA  ->  OpenCL (OpenCV T-API)  ->  CPU

CUDA uses the dedicated cv2.dnn CUDA backend; OpenCL uses the OpenCV backend with
the OpenCL target and the T-API (cv2.UMat) so colour conversion / Haar detection
run on the GPU; CPU is the always-available fallback.

``select_acceleration`` is a pure function of (priority, capabilities) so it can
be unit-tested on a CI runner with no GPU. ``apply_to_net`` / ``to_umat`` wire the
chosen mode into an actual ``cv2.dnn.Net`` and frame at runtime.
"""

import logging

import cv2

import config

logger = logging.getLogger(__name__)

CUDA = "CUDA"
OPENCL = "OpenCL"
CPU = "CPU"


class Acceleration:
    """A resolved acceleration choice: a name plus its cv2.dnn backend/target."""

    def __init__(self, name, backend, target):
        self.name = name
        self.backend = backend
        self.target = target

    def __repr__(self):
        return f"Acceleration(name={self.name!r})"

    def __eq__(self, other):
        return isinstance(other, Acceleration) and self.name == other.name


def _const(name, default=None):
    """Fetch a cv2.dnn constant defensively (older/headless builds may lack it)."""
    return getattr(cv2.dnn, name, default)


def probe_capabilities():
    """Detect available accelerators. Every probe is guarded so an exotic or
    headless OpenCV build degrades cleanly to CPU instead of raising."""
    caps = {"cuda": False, "cuda_targets": [], "opencl": False}
    try:
        caps["cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (cv2.error, AttributeError) as e:
        logger.debug("CUDA probe unavailable: %s", e)
    try:
        backends = cv2.dnn.getAvailableBackends()
        if _const("DNN_BACKEND_CUDA") in backends:
            caps["cuda_targets"] = list(
                cv2.dnn.getAvailableTargets(_const("DNN_BACKEND_CUDA"))
            )
    except (cv2.error, AttributeError) as e:
        logger.debug("CUDA target probe unavailable: %s", e)
    try:
        caps["opencl"] = bool(cv2.ocl.haveOpenCL())
    except (cv2.error, AttributeError) as e:
        logger.debug("OpenCL probe unavailable: %s", e)
    return caps


def select_acceleration(priority=None, caps=None):
    """Return the first viable Acceleration given a priority list + capabilities.

    Pure/deterministic: pass ``caps`` to test without real hardware. Falls back
    to CPU if the priority list yields nothing usable.
    """
    priority = priority if priority is not None else config.ACCELERATION_PRIORITY
    caps = caps if caps is not None else probe_capabilities()

    cpu = Acceleration(CPU, _const("DNN_BACKEND_OPENCV"), _const("DNN_TARGET_CPU"))

    for name in priority:
        if name == CUDA and caps.get("cuda") and \
                _const("DNN_TARGET_CUDA") in caps.get("cuda_targets", []):
            return Acceleration(CUDA, _const("DNN_BACKEND_CUDA"), _const("DNN_TARGET_CUDA"))
        if name == OPENCL and caps.get("opencl"):
            return Acceleration(OPENCL, _const("DNN_BACKEND_OPENCV"), _const("DNN_TARGET_OPENCL"))
        if name == CPU:
            return cpu
    return cpu


def apply_to_net(net, accel):
    """Configure a cv2.dnn.Net for the chosen acceleration and return ``accel``.

    Toggles the global OpenCL switch to match the selection so the T-API path is
    only active when OpenCL was actually chosen.
    """
    try:
        cv2.ocl.setUseOpenCL(accel.name == OPENCL)
    except (cv2.error, AttributeError) as e:
        logger.debug("Could not toggle OpenCL: %s", e)
    net.setPreferableBackend(accel.backend)
    net.setPreferableTarget(accel.target)
    logger.info("%s acceleration enabled (target=%s)", accel.name, accel.target)
    return accel


def to_umat(frame, accel):
    """Wrap a frame in a cv2.UMat when OpenCL is active so downstream OpenCV ops
    execute on the GPU via the T-API. No-op (returns the frame) otherwise."""
    try:
        if accel is not None and accel.name == OPENCL and cv2.ocl.useOpenCL():
            return cv2.UMat(frame)
    except Exception as e:
        logger.warning("UMat conversion failed, falling back to CPU: %s", e)
    return frame
