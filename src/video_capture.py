"""
Video Capture Module for Face Tracking
Author: Romil V. Shah
This module handles video capture operations for the face tracking application.
"""

import cv2
import time
from typing import Tuple, Optional

class VideoCapture:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_frame_time = 0

    def start(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            self.last_frame_time = time.time()
            return True
        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[object]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None

        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)

        success, frame = self.cap.read()
        if not success:
            print("Frame grab failed, trying again...")
            success, frame = self.cap.read()

        self.last_frame_time = time.time()
        return success, frame

    def stop(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
