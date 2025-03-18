"""
Video capture module for the face tracking application.
"""

import cv2
import time
from typing import Tuple, Optional

class VideoCapture:
    """
    Class for capturing video from camera.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize the video capture with the given parameters.
        
        Args:
            camera_index: Index of the camera to use
            width: Width of the captured frame
            height: Height of the captured frame
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_time = 1.0 / fps if fps > 0 else 0
        self.last_frame_time = 0
        
    def start(self) -> bool:
        """
        Start the video capture.
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Try to use DirectShow backend first (works better on Windows)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                print(f"DirectShow backend failed, trying default backend...")
                # If DirectShow fails, try the default backend
                self.cap = cv2.VideoCapture(self.camera_index)
                
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set MJPG format (helps with some webcams)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self.last_frame_time = time.time()
            return True
        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[object]]:
        """
        Read a frame from the video capture with frame rate control.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        # Implement basic frame rate control
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.frame_time:
            # Sleep to maintain desired frame rate
            time.sleep(self.frame_time - elapsed)
        
        success, frame = self.cap.read()
        
        if not success:
            # If frame grab failed, try again once
            print("Frame grab failed, trying again...")
            success, frame = self.cap.read()
        
        self.last_frame_time = time.time()
        
        return success, frame
    
    def stop(self):
        """
        Stop the video capture.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
