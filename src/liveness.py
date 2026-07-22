"""Blink-based liveness detection using Eye Aspect Ratio (EAR).

Mediapipe is chosen over dlib (requires CMake/Boost native compile) and
OpenCV LBF (requires a separate model download) because it bundles its
TFLite models with no native build step and is Apache 2.0 licensed.

Environment
-----------
LIVENESS_CHECK_ENABLED : str (default "false")
    Set to "true", "1", or "yes" to enable the liveness detector in api.py.
    The REST /detect endpoint always returns checked=false (single-frame);
    real blink detection only runs on frame sequences (webcam path).

Supported Python: 3.8–3.12 (mediapipe does not support 3.13+ yet).
"""

import os

import numpy as np

LIVENESS_CHECK_ENABLED: bool = os.environ.get(
    "LIVENESS_CHECK_ENABLED", "false"
).lower() in ("1", "true", "yes")

_MEDIAPIPE_AVAILABLE = False
_face_mesh = None

try:
    import mediapipe as mp  # type: ignore[import]

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore[assignment]


def _get_face_mesh():
    global _face_mesh
    if not _MEDIAPIPE_AVAILABLE:
        return None
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )
    return _face_mesh


def _ear(landmarks, indices, img_w, img_h):
    """Eye Aspect Ratio from six mediapipe landmark indices."""
    pts = [
        (landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices
    ]
    # vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal distance
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h + 1e-6)


_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
_LEFT_EYE = [362, 385, 387, 263, 373, 380]


class LivenessDetector:
    """Wrapper around mediapipe-based EAR blink detection."""

    def check_single_image(self, frame: np.ndarray) -> dict:
        """REST path: blink detection requires a frame sequence, not a single image."""
        return {"checked": False, "reason": "single_frame_input"}

    def check_frame_sequence(self, frames) -> dict:
        """Webcam path: compute EAR std-dev across a sequence of frames.

        Returns checked=True with a liveness flag and confidence if mediapipe
        is available and at least 5 frames are provided.
        """
        if not _MEDIAPIPE_AVAILABLE:
            return {"checked": False, "reason": "mediapipe_unavailable"}
        if len(frames) < 5:
            return {"checked": False, "reason": "insufficient_frames"}

        mesh = _get_face_mesh()
        ear_values = []
        for frame in frames:
            h, w = frame.shape[:2]
            rgb = frame[:, :, ::-1]  # BGR → RGB
            result = mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                ear = (_ear(lm, _RIGHT_EYE, w, h) + _ear(lm, _LEFT_EYE, w, h)) / 2.0
                ear_values.append(ear)

        if not ear_values:
            return {"checked": False, "reason": "no_face_detected"}

        ear_std = float(np.std(ear_values))
        live = ear_std > 0.02
        confidence = round(min(ear_std / 0.04, 1.0), 4)
        return {"checked": True, "live": live, "confidence": confidence}
