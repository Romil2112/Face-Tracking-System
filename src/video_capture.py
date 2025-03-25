"""
Video Capture Module for Face Tracking
Author: Romil V. Shah
This module handles video capture operations with robust error recovery and resource management.
"""

import cv2
import time
import logging
from typing import Tuple, Optional
import numpy as np
import config
from error_handling import retry, ErrorHandler

logger = logging.getLogger(__name__)

class VideoCapture:
    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        """Initialize video capture with configuration parameters."""
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.cap = None
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_frame_time = 0
        self._is_running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __del__(self):
        self.stop()

    @retry(max_attempts=3, delay=1, allowed_exceptions=(IOError, cv2.error))
    def start(self) -> bool:
        """Initialize video capture device with error handling."""
        try:
            # Try multiple backend APIs for compatibility
            for api in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                self.cap = cv2.VideoCapture(self.camera_index, api)
                if self.cap.isOpened():
                    break
            if not self.cap.isOpened():
                raise IOError(f"Could not open camera {self.camera_index}")

            # Set camera properties with validation
            self._set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._set_camera_property(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._set_camera_property(cv2.CAP_PROP_FPS, self.target_fps)

            # Verify actual frame parameters
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_width, actual_height) != (self.target_width, self.target_height):
                logger.warning(f"Camera resolution mismatch: "
                               f"Requested ({self.target_width}x{self.target_height}), "
                               f"Actual ({actual_width}x{actual_height})")

            self._is_running = True
            return True
        except Exception as e:
            logger.error(f"Video capture initialization failed: {str(e)}")
            ErrorHandler.handle_camera_error(e)
            self.stop()
            return False

    def _set_camera_property(self, prop_id: int, value: float) -> None:
        """Set camera property with validation."""
        if not self.cap.set(prop_id, value):
            logger.warning(f"Failed to set camera property {prop_id} to {value}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with error recovery and timing control."""
        try:
            if not self._is_running or not self.cap.isOpened():
                return False, None

            # Non-blocking frame acquisition
            self.cap.grab()

            # Maintain frame rate timing
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)

            # Retrieve the frame if it's time to process
            if config.DYNAMIC_FRAME_SKIP:
                success, frame = self.cap.retrieve()
            else:
                success, frame = False, None

            self.last_frame_time = time.time()

            if not success or frame is None or frame.size == 0:
                logger.warning("Frame read failed, attempting recovery...")
                self._recover_capture()
                return False, None

            return True, frame
        except Exception as e:
            logger.error(f"Critical read error: {str(e)}")
            ErrorHandler.handle_camera_error(e)
            self._recover_capture()
            return False, None

    def _recover_capture(self) -> None:
        """Attempt to recover from capture failures."""
        self.stop()
        time.sleep(0.5)
        try:
            self.start()
        except Exception as e:
            logger.error(f"Capture recovery failed: {str(e)}")

    def stop(self) -> None:
        """Safely release video resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Check if capture is active."""
        return self._is_running and self.cap is not None and self.cap.isOpened()
