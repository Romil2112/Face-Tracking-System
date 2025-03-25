"""
Configuration Module for Face Tracking
Author: Romil V. Shah
This module contains configuration parameters for the face tracking application.
"""

import os

# Camera config
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Face detection
CASCADE_PATH = os.path.join(os.path.dirname(__file__), '..', 'haarcascade_frontalface_default.xml')
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 4
MIN_SIZE = (30, 30)

# DNN face detection
DNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
DNN_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models', 'deploy.prototxt')
DNN_CONFIDENCE_THRESHOLD = 0.7 # Reduce false positives
DNN_INPUT_SIZE = (300, 300)

# Haar Cascade
HAAR_SCALE_FACTOR = 1.2 # Faster pyramid scaling

# Visualization
FACE_RECT_COLOR = (0, 255, 0)
FACE_RECT_THICKNESS = 2
FACE_CENTER_COLOR = (0, 0, 255)
FACE_CENTER_RADIUS = 5
FONT = 'FONT_HERSHEY_SIMPLEX'
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 2

# Eye detection
EYE_DETECTION_ENABLED = True
EYE_MIN_NEIGHBORS = 2
EYE_SCALE_FACTOR = 1.1
EYE_MIN_SIZE = (15, 15)

# Face geometry
FACE_ASPECT_RATIO_MIN = 0.5
FACE_ASPECT_RATIO_MAX = 1.1
MINIMUM_CONFIDENCE = 0.4

# Temporal filtering
TEMPORAL_FILTERING_ENABLED = True
TEMPORAL_FRAMES_HISTORY = 5
TEMPORAL_CONSISTENCY = 2 # Allow brief tracking gaps
TEMPORAL_CONSISTENCY_THRESHOLD = 2 # Must match TEMPORAL_CONSISTENCY value

# Motion detection
MOTION_DETECTION_ENABLED = False
MOTION_THRESHOLD = 5.0

# Performance
MAX_FACES = 10
DYNAMIC_FRAME_SKIP = True # Auto-adjust based on load
MAX_FRAME_SKIP = 5 # Maximum allowed frame skip
FRAME_SKIP = 1 # Default number of frames to skip between processing

# Debug options
DEBUG_MODE = True

# NMS
NMS_THRESHOLD = 0.4

# Acceleration priority
ACCELERATION_PRIORITY = ["CUDA", "CPU"]  # Explicitly exclude OpenCL
