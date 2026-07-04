"""
Face Tracking Application - Main Module
Author: Romil V. Shah
This module contains the main execution logic for the face tracking application.
"""

import argparse
import concurrent.futures
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from typing import Any

import cv2
import numpy as np
from imutils.video import FPS

import acceleration
import config
from error_handling import ErrorHandler, retry
from face_detector import FaceDetector
from nms_utils import apply_nms
from tracking_visualizer import TrackingVisualizer
from video_capture import VideoCapture

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment for optimal performance. Defaults only (setdefault) so
# the user can override the OpenCL device; prefer any GPU rather than pinning a
# specific vendor.
# OpenCL kernel cache dir: configurable, and defaults under the platform temp
# dir rather than a hardcoded world-writable /tmp path.
_ocl_cache = os.environ.get("OPENCV_OCL_CACHE_DIR") or os.path.join(
    tempfile.gettempdir(), "ocl_cache"
)
os.environ.setdefault("OPENCV_OCL4DNN_CONFIG_PATH", _ocl_cache)
os.environ.setdefault("OPENCV_OPENCL_DEVICE", ":GPU:0")  # any platform, first GPU
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

def parse_arguments() -> dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time face tracking using OpenCV')
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX,
                       help=f'Camera index (default: {config.CAMERA_INDEX})')
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH,
                       help=f'Camera width (default: {config.CAMERA_WIDTH})')
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT,
                       help=f'Camera height (default: {config.CAMERA_HEIGHT})')
    parser.add_argument('--cascade', type=str, default=config.CASCADE_PATH,
                       help=f'Path to Haar cascade XML file (default: {config.CASCADE_PATH})')
    parser.add_argument('--scale-factor', type=float, default=config.SCALE_FACTOR,
                       help=f'Scale factor for face detection (default: {config.SCALE_FACTOR})')
    parser.add_argument('--min-neighbors', type=int, default=config.MIN_NEIGHBORS,
                       help=f'Min neighbors for face detection (default: {config.MIN_NEIGHBORS})')
    parser.add_argument('--max-faces', type=int, default=config.MAX_FACES,
                       help=f'Maximum number of faces to track (default: {config.MAX_FACES})')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    return vars(parser.parse_args())

@retry(max_attempts=3, delay=2, allowed_exceptions=(IOError, cv2.error))
def initialize_video_capture(camera_index: int, width: int, height: int, fps: int) -> VideoCapture:
    """Initialize and configure video capture with retry logic."""
    video_capture = VideoCapture(camera_index, width, height, fps)
    if not video_capture.start():
        raise OSError("Failed to start video capture")
    return video_capture

def verify_acceleration():
    """Report the acceleration that will actually be used (CUDA/OpenCL/CPU).

    Returns True when a GPU path (CUDA or OpenCL) is selected, False for CPU.
    """
    try:
        caps = acceleration.probe_capabilities()
        accel = acceleration.select_acceleration(caps=caps)
        logger.info(
            "Acceleration selected: %s (CUDA available=%s, OpenCL available=%s)",
            accel.name, caps["cuda"], caps["opencl"],
        )
        return accel.name in (acceleration.CUDA, acceleration.OPENCL)
    except Exception as e:
        logger.error(f"Acceleration verification failed: {str(e)}")
        return False

def validate_configuration() -> None:
    """Validate critical configuration parameters."""
    required_files = [
        (config.CASCADE_PATH, "Haar Cascade XML"),
        (config.DNN_MODEL_PATH, "DNN Model"),
        (config.DNN_CONFIG_PATH, "DNN Config")
    ]

    for path, name in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")

    if not 0 < config.DNN_CONFIDENCE_THRESHOLD <= 1:
        raise ValueError("DNN confidence threshold must be between 0 and 1")

    if config.CAMERA_INDEX < 0:
        raise ValueError("Camera index cannot be negative")

    _validate_cuda_backend()


def _validate_cuda_backend() -> None:
    """If a CUDA backend is reported, require that CUDA targets exist too."""
    try:
        available_backends = cv2.dnn.getAvailableBackends()
    except AttributeError:
        available_backends = ["OPENCV", "CUDA"]

    if cv2.dnn.DNN_BACKEND_CUDA in available_backends:
        if cv2.dnn.DNN_TARGET_CUDA not in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA):
            raise RuntimeError("CUDA backend available but CUDA targets missing")

def _detect_and_filter(frame: np.ndarray, face_detector: FaceDetector,
                       args: dict[str, Any]) -> list[dict]:
    """Run DNN + Haar in parallel, merge, NMS, confidence-filter, cap at max_faces."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        dnn_future = executor.submit(face_detector.detect_faces_dnn, frame)
        haar_future = executor.submit(face_detector.detect_faces_haar, frame)
        dnn_faces = dnn_future.result()
        haar_faces = haar_future.result()

    faces = apply_nms(dnn_faces + haar_faces, config.NMS_THRESHOLD)
    faces = [f for f in faces if f['confidence'] >= config.MINIMUM_CONFIDENCE]
    return faces[:args['max_faces']]


def _apply_motion_gate(faces: list[dict], prev_frame: np.ndarray,
                       frame: np.ndarray) -> list[dict]:
    """Keep only faces whose region shows motion above config.MOTION_THRESHOLD."""
    try:
        from motion_utils import detect_motion
        motion_magnitude = detect_motion(prev_frame, frame)
        if motion_magnitude is None:
            return faces
        motion_faces = []
        for face in faces:
            x, y, w, h = face['rect']
            face_motion = np.mean(motion_magnitude[y:y+h, x:x+w])
            if face_motion >= config.MOTION_THRESHOLD:
                motion_faces.append(face)
        return motion_faces
    except Exception as e:
        if config.DEBUG_MODE:
            logger.debug(f"Motion detection error: {e}")
        return faces


def _apply_temporal(faces: list[dict], temporal_filter, prev_frame: np.ndarray) -> list[dict]:
    """Run the temporal-consistency filter, deriving a safe frame time-delta."""
    try:
        current_time = datetime.now()
        if prev_frame is not None and isinstance(prev_frame, np.ndarray):
            time_diff = current_time - getattr(prev_frame, 'timestamp', current_time)
            if time_diff.total_seconds() <= 0:
                time_diff = timedelta(milliseconds=33)  # Default to 30 FPS
        else:
            time_diff = timedelta(milliseconds=33)
        return temporal_filter.update(faces, time_diff)
    except Exception as e:
        if config.DEBUG_MODE:
            logger.debug(f"Temporal filtering error: {e}")
        return faces


def process_frame(frame: np.ndarray, face_detector: FaceDetector, args: dict[str, Any],
                 motion_detection: bool, prev_frame: np.ndarray, temporal_filter) -> list[dict]:
    """Process a single frame through the detection pipeline."""
    try:
        faces = _detect_and_filter(frame, face_detector, args)
        if config.DEBUG_MODE:
            logger.debug(f"Detected {len(faces)} faces after NMS")
        if config.MOTION_DETECTION_ENABLED and prev_frame is not None:
            faces = _apply_motion_gate(faces, prev_frame, frame)
        if temporal_filter is not None:
            faces = _apply_temporal(faces, temporal_filter, prev_frame)
        return faces
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return []

def _next_frame(video_capture: VideoCapture, retry_count: int, max_retries: int):
    """Read the next frame. Returns (frame_or_None, retry_count).

    On a good read, retry_count resets to 0. On a bad read it increments; once it
    hits max_retries an IOError is raised so the loop stops.
    """
    success, frame = video_capture.read()
    if success and frame is not None:
        return frame, 0
    retry_count += 1
    if retry_count >= max_retries:
        raise OSError("Maximum retry attempts reached. Stopping face tracking.")
    time.sleep(1)
    return None, retry_count


def _detect_with_reset(frame, face_detector, args, motion_detection,
                       prev_frame, temporal_filter, prev_faces):
    """process_frame, recovering from the known NoneType/datetime TypeError by
    resetting the temporal filter and keeping the previous frame's faces."""
    try:
        return process_frame(frame, face_detector, args,
                             motion_detection, prev_frame, temporal_filter)
    except TypeError as e:
        if "unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'" in str(e):
            logger.warning("Encountered datetime operation error, resetting temporal filter")
            temporal_filter.reset()
            return prev_faces
        raise


def _render(visualizer: TrackingVisualizer, frame, faces: list[dict], fps) -> None:
    """Update the FPS counter and show the annotated frame."""
    fps.update()
    fps.stop()
    output_frame = visualizer.draw_faces(frame, faces, fps.fps())
    cv2.imshow('Face Tracking', output_frame)


def _should_quit() -> bool:
    """True when the user pressed 'q' or closed the window."""
    return (cv2.waitKey(1) & 0xFF == ord('q')
            or cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_VISIBLE) < 1)


def _handle_loop_error(error: Exception, error_handler: ErrorHandler) -> None:
    """Log a main-loop error and re-raise it unless the handler recovers."""
    logger.error(f"Error in main loop: {error}")
    if not error_handler.handle_camera_error(error):
        raise error


def main_loop(video_capture: VideoCapture, face_detector: FaceDetector,
             visualizer: TrackingVisualizer, args: dict[str, Any],
             motion_detection: bool, temporal_filter, error_handler: ErrorHandler) -> None:
    """Main processing loop with error handling and FPS integration."""
    process_this_frame = 0
    prev_frame = None
    retry_count = 0
    max_retries = 3
    faces: list[dict] = []
    fps = FPS().start()

    try:
        while True:
            frame, retry_count = _next_frame(video_capture, retry_count, max_retries)
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)

            if process_this_frame == 0:
                faces = _detect_with_reset(frame, face_detector, args,
                                           motion_detection, prev_frame, temporal_filter, faces)
                if motion_detection:
                    prev_frame = frame.copy()
                    prev_frame.timestamp = datetime.now()

            _render(visualizer, frame, faces, fps)
            process_this_frame = (process_this_frame + 1) % config.FRAME_SKIP

            if _should_quit():
                break

    except Exception as e:
        _handle_loop_error(e, error_handler)
    finally:
        fps.stop()
        video_capture.stop()
        cv2.destroyAllWindows()
        logger.info(f"Final FPS: {fps.fps():.2f}")

def main() -> None:
    """Main application entry point."""
    args = parse_arguments()
    error_handler = ErrorHandler()

    try:
        validate_configuration()

        if args['debug']:
            config.DEBUG_MODE = True
            logging.basicConfig(level=logging.DEBUG)

        # Initialize components
        face_detector = FaceDetector(
            dnn_model_path=config.DNN_MODEL_PATH,
            dnn_config_path=config.DNN_CONFIG_PATH,
            cascade_path=config.CASCADE_PATH,
            scale_factor=args['scale_factor'],
            min_neighbors=args['min_neighbors'],
            min_size=config.MIN_SIZE
        )

        video_capture = initialize_video_capture(
            args['camera'],
            args['width'],
            args['height'],
            config.CAMERA_FPS
        )

        # Verify acceleration support
        verify_acceleration()

        # Initialize temporal filter
        temporal_filter = None
        if config.TEMPORAL_FILTERING_ENABLED:
            try:
                from temporal_filter import TemporalFilter
                temporal_filter = TemporalFilter(
                    history_size=config.TEMPORAL_FRAMES_HISTORY,
                    consistency_threshold=config.TEMPORAL_CONSISTENCY_THRESHOLD
                )
            except Exception as e:
                logger.warning(f"Temporal filtering disabled: {e}")

        # Initialize motion detection
        motion_detection = config.MOTION_DETECTION_ENABLED
        if motion_detection:
            try:
                from motion_utils import detect_motion  # noqa: F401  # availability probe
            except ImportError as e:
                motion_detection = False
                logger.warning(f"Motion detection disabled: {e}")

        # Initialize visualizer
        visualizer = TrackingVisualizer(
            rect_color=config.FACE_RECT_COLOR,
            rect_thickness=config.FACE_RECT_THICKNESS,
            center_color=config.FACE_CENTER_COLOR,
            center_radius=config.FACE_CENTER_RADIUS,
            font=getattr(cv2, config.FONT),
            font_scale=config.FONT_SCALE,
            font_color=config.FONT_COLOR,
            font_thickness=config.FONT_THICKNESS
        )

        main_loop(video_capture, face_detector, visualizer, args,
                 motion_detection, temporal_filter, error_handler)

    except Exception as e:
        logger.critical(f"Critical error: {e}")
        sys.exit(1)
    finally:
        logger.info("Face tracking stopped")

if __name__ == '__main__':
    main()
