"""
Face Tracking Application - Main Module
Author: Romil V. Shah
This module contains the main execution logic for the face tracking application.
"""

import cv2
import time
import argparse
import sys
import os
import numpy as np
import concurrent.futures
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from imutils.video import FPS
from face_detector import FaceDetector
from video_capture import VideoCapture
from tracking_visualizer import TrackingVisualizer
from error_handling import retry, ErrorHandler
from nms_utils import apply_nms
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure environment for optimal performance
os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "/tmp/ocl_cache"  # Linux/macOS
os.environ["OPENCV_OPENCL_DEVICE"] = "AMD:GPU"  # Force specific device
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def parse_arguments() -> Dict[str, Any]:
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
        raise IOError("Failed to start video capture")
    return video_capture

def verify_acceleration():
    """Validate hardware acceleration availability"""
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cuda_backend = getattr(cv2.dnn, 'DNN_BACKEND_CUDA', cv2.dnn.DNN_BACKEND_OPENCV)
            available_targets = getattr(cv2.dnn, 'getAvailableTargets', lambda x: [])(cuda_backend)
            if getattr(cv2.dnn, 'DNN_TARGET_CUDA', None) not in available_targets:
                logger.warning(f"CUDA targets unavailable for backend {cuda_backend}")
            logger.info(f"CUDA acceleration available (Targets: {available_targets})")
            return True
        else:
            cpu_backend = cv2.dnn.DNN_BACKEND_OPENCV
            available_targets = getattr(cv2.dnn, 'getAvailableTargets', lambda x: [])(cpu_backend)
            logger.info(f"CPU acceleration active (Available targets: {available_targets})")
            return False
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
    
    # Add backend validation
    available_backends = []
    try:
        available_backends = cv2.dnn.getAvailableBackends()
    except AttributeError:
        available_backends = ["OPENCV", "CUDA"]
    
    if cv2.dnn.DNN_BACKEND_CUDA in available_backends:
        if cv2.dnn.DNN_TARGET_CUDA not in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA):
            raise RuntimeError("CUDA backend available but CUDA targets missing")

def process_frame(frame: np.ndarray, face_detector: FaceDetector, args: Dict[str, Any],
                 motion_detection: bool, prev_frame: np.ndarray, temporal_filter) -> List[Dict]:
    """Process a single frame through the detection pipeline."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dnn_future = executor.submit(face_detector.detect_faces_dnn, frame)
            haar_future = executor.submit(face_detector.detect_faces_haar, frame)
            dnn_faces = dnn_future.result()
            haar_faces = haar_future.result()
        
        combined_faces = dnn_faces + haar_faces
        faces = apply_nms(combined_faces, config.NMS_THRESHOLD)
        faces = [f for f in faces if f['confidence'] >= config.MINIMUM_CONFIDENCE]
        faces = faces[:args['max_faces']]
        
        if config.DEBUG_MODE:
            logger.debug(f"Detected {len(faces)} faces after NMS")
        
        if config.MOTION_DETECTION_ENABLED and prev_frame is not None:
            try:
                from motion_utils import detect_motion
                motion_magnitude = detect_motion(prev_frame, frame)
                if motion_magnitude is not None:
                    motion_faces = []
                    for face in faces:
                        x, y, w, h = face['rect']
                        face_motion = np.mean(motion_magnitude[y:y+h, x:x+w])
                        if face_motion >= config.MOTION_THRESHOLD:
                            motion_faces.append(face)
                    faces = motion_faces
            except Exception as e:
                if config.DEBUG_MODE:
                    logger.debug(f"Motion detection error: {e}")
        
        if temporal_filter is not None:
            try:
                current_time = datetime.now()
                if prev_frame is not None and isinstance(prev_frame, np.ndarray):
                    # Ensure valid time difference
                    time_diff = current_time - getattr(prev_frame, 'timestamp', current_time)
                    if time_diff.total_seconds() <= 0:
                        time_diff = timedelta(milliseconds=33)  # Default to 30 FPS
                else:
                    time_diff = timedelta(milliseconds=33)
                
                faces = temporal_filter.update(faces, time_diff)
            except Exception as e:
                if config.DEBUG_MODE:
                    logger.debug(f"Temporal filtering error: {e}")
        
        return faces
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return []

def main_loop(video_capture: VideoCapture, face_detector: FaceDetector,
             visualizer: TrackingVisualizer, args: Dict[str, Any],
             motion_detection: bool, temporal_filter, error_handler: ErrorHandler) -> None:
    """Main processing loop with error handling and FPS integration."""
    process_this_frame = 0
    prev_frame = None
    retry_count = 0
    max_retries = 3
    fps = FPS().start()

    try:
        while True:
            success, frame = video_capture.read()
            if not success or frame is None:
                retry_count += 1
                if retry_count >= max_retries:
                    raise IOError("Maximum retry attempts reached. Stopping face tracking.")
                time.sleep(1)
                continue
            
            retry_count = 0
            frame = cv2.flip(frame, 1)
            
            if process_this_frame == 0:
                try:
                    faces = process_frame(frame, face_detector, args,
                                        motion_detection, prev_frame, temporal_filter)
                except TypeError as e:
                    if "unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'" in str(e):
                        logger.warning("Encountered datetime operation error, resetting temporal filter")
                        temporal_filter.reset()  # Implement a reset method in your TemporalFilter class
                    else:
                        raise
                
                if motion_detection:
                    prev_frame = frame.copy()
                    prev_frame.timestamp = datetime.now()
            
            # Update FPS counter and get current value
            fps.update()
            fps.stop()
            current_fps = fps.fps()
            
            # Generate output frame with visualizations
            output_frame = visualizer.draw_faces(frame, faces, current_fps)
            cv2.imshow('Face Tracking', output_frame)
            
            process_this_frame = (process_this_frame + 1) % config.FRAME_SKIP
            
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        if not error_handler.handle_camera_error(e):
            raise
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
                from motion_utils import detect_motion
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
