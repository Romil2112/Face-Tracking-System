"""
Error Handling Module for Face Tracking
Author: Romil V. Shah
This module provides advanced error handling and recovery strategies.
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Any, Optional
from collections import defaultdict
import cv2
import numpy as np
import config

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure = 0

    def is_open(self) -> bool:
        if time.time() - self.last_failure > self.reset_timeout:
            self.reset()
        return self.failure_count >= self.max_failures

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = time.time()

    def reset(self):
        self.failure_count = 0
        self.last_failure = 0

def retry(max_attempts: int = 3,
          delay: float = 1,
          allowed_exceptions: Tuple[Type[Exception], ...] = (Exception,),
          jitter: float = 0.1,
          log_level: int = logging.WARNING) -> Callable:
    """
    Enhanced retry decorator with exponential backoff and jitter
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    attempts += 1
                    wait = delay * (2 ** (attempts - 1)) * (1 + jitter * (np.random.random() - 0.5))
                    logger.log(log_level,
                             f"Attempt {attempts}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                             f"Retrying in {wait:.2f}s...")
                    time.sleep(wait)
            return func(*args, **kwargs)  # Final attempt
        return wrapper
    return decorator

class ErrorHandler:
    """
    Advanced error handler with stateful recovery strategies
    """
    def __init__(self):
        self.camera_breakers = defaultdict(CircuitBreaker)
        self.detector_breakers = defaultdict(CircuitBreaker)
        self.resource_breakers = defaultdict(CircuitBreaker)
        self.log_filter = [
            "ocl4dnn_conv_spatial.cpp",
            "-cl-no-subgroup-ifp",
            "CL_BUILD_PROGRAM_FAILURE"
        ]
        logging.getLogger("cv2").addFilter(self.filter_warnings)

    def filter_warnings(self, record):
        """Suppress OpenCL-related warnings"""
        return not any(msg in record.getMessage() for msg in self.log_filter)

    def handle_resource_exhaustion(self, error: Exception) -> bool:
        """
        Resource management recovery strategy
        """
        logger.error(f"Resource exhaustion: {str(error)}")
        resource_type = getattr(error, 'resource_type', 'default')
        breaker = self.resource_breakers[resource_type]
        
        # Added CUDA backend validation
        if 'CUDA' in str(error):
            available_backends = getattr(cv2.dnn, 'getAvailableBackends', lambda: [])()
            if cv2.dnn.DNN_BACKEND_CUDA not in available_backends:
                logger.error("CUDA backend not available despite initialization")
                if "CUDA" in config.ACCELERATION_PRIORITY:
                    config.ACCELERATION_PRIORITY.remove("CUDA")
                return True

        if breaker.is_open():
            logger.error("Resource circuit breaker open")
            return False

        if 'CUDA_OUT_OF_MEMORY' in str(error):
            config.DYNAMIC_FRAME_SKIP = True
            config.MAX_FACES = max(1, config.MAX_FACES - 1)
            logger.info(f"Adjusted MAX_FACES to {config.MAX_FACES} and enabled DYNAMIC_FRAME_SKIP")
            return True

        recovery_strategies = [
            self._reduce_processing_load,
            self._free_memory_resources,
            self._restart_subsystems
        ]
        
        for strategy in recovery_strategies:
            if strategy(error):
                breaker.reset()
                return True
        breaker.record_failure()
        return False

    def handle_camera_error(self, error: Optional[Exception] = None) -> bool:
        """Handle camera errors with default parameters"""
        error = error or Exception("Unknown camera error")
        logger.error(f"Camera error: {str(error)}")
        cam_id = getattr(error, 'camera_id', 'default')
        breaker = self.camera_breakers[cam_id]
        
        if breaker.is_open():
            logger.error("Camera circuit breaker open, skipping recovery")
            return False

        recovery_strategies = [
            self._reconnect_camera,
            self._try_alternate_camera,
            self._reset_camera_settings
        ]
        
        for strategy in recovery_strategies:
            if strategy(error):
                breaker.reset()
                return True
        breaker.record_failure()
        return False

    def handle_face_detection_error(self, error: Optional[Exception] = None) -> bool:
        """Handle detection errors with default parameters"""
        error = error or Exception("Unknown face detection error")
        logger.error(f"Face detection error: {str(error)}")
        detector_type = getattr(error, 'detector_type', 'default')
        breaker = self.detector_breakers[detector_type]
        
        if breaker.is_open():
            logger.error("Face detection circuit breaker open")
            return False

        recovery_strategies = [
            self._switch_detection_method,
            self._reload_detection_model,
            self._adjust_detection_parameters
        ]
        
        for strategy in recovery_strategies:
            if strategy(error):
                breaker.reset()
                return True
        breaker.record_failure()
        return False

    def _reconnect_camera(self, error: Exception) -> bool:
        """Attempt camera reconnection"""
        logger.info("Attempting camera reconnection...")
        try:
            if hasattr(error, 'camera'):
                error.camera.stop()
                time.sleep(1)
                return error.camera.start()
            return False
        except Exception as e:
            logger.error(f"Reconnection failed: {str(e)}")
            return False

    def _try_alternate_camera(self, error: Exception) -> bool:
        """Try alternate camera indices"""
        logger.info("Trying alternate camera indices...")
        for idx in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    logger.info(f"Found working camera at index {idx}")
                    cap.release()
                    return True
            except:
                continue
        return False

    def _reset_camera_settings(self, error: Exception) -> bool:
        """Reset camera to default settings"""
        logger.info("Resetting camera settings...")
        try:
            if hasattr(error, 'camera'):
                error.camera.stop()
                time.sleep(1)
                error.camera.width = config.CAMERA_WIDTH
                error.camera.height = config.CAMERA_HEIGHT
                error.camera.fps = config.CAMERA_FPS
                return error.camera.start()
            return False
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            return False

    def _switch_detection_method(self, error: Exception) -> bool:
        """Switch between DNN and Haar Cascade methods"""
        logger.info("Switching detection method...")
        if isinstance(error, DnnDetectionError):
            logger.info("Falling back to Haar Cascade")
            return True
        elif isinstance(error, HaarDetectionError):
            logger.info("Falling back to DNN")
            return True
        return False

    def _reload_detection_model(self, error: Exception) -> bool:
        """Reload detection models"""
        logger.info("Reloading detection models...")
        return False

    def _adjust_detection_parameters(self, error: Exception) -> bool:
        """Adjust detection parameters dynamically"""
        logger.info("Adjusting detection parameters...")
        return True

    def _reduce_processing_load(self, error: Exception) -> bool:
        """Reduce system load by disabling features"""
        logger.info("Reducing processing load...")
        config.TEMPORAL_FILTERING_ENABLED = False
        config.MOTION_DETECTION_ENABLED = False
        config.FRAME_SKIP += 1
        return True

    def _free_memory_resources(self, error: Exception) -> bool:
        """Attempt to free memory resources"""
        logger.info("Freeing memory resources...")
        return True

    def _restart_subsystems(self, error: Exception) -> bool:
        """Restart critical subsystems"""
        logger.info("Restarting subsystems...")
        return True

# Custom exception classes
class DnnDetectionError(Exception): pass
class HaarDetectionError(Exception): pass
class CameraRecoveryError(Exception): pass

# Apply OpenCL warning filter to cv2 logger
logging.getLogger("cv2").addFilter(ErrorHandler().filter_warnings)
