"""Configuration for face tracking app."""

# Camera config
CAMERA_INDEX = 0  # Default camera
CAMERA_WIDTH = 640  # Frame width
CAMERA_HEIGHT = 480  # Frame height
CAMERA_FPS = 30  # Target framerate

# Camera backend
CAMERA_BACKEND_DSHOW = True  # DirectShow (Windows)
CAMERA_BACKEND_MSMF = False  # Media Foundation
CAMERA_FOURCC = 'MJPG'  # Video compression format

# Face detection - Less strict parameters
CASCADE_PATH = 'haarcascade_frontalface_default.xml'  # Haar cascade file
SCALE_FACTOR = 1.1  # Image scaling factor
MIN_NEIGHBORS = 4  # Reduced for better detection
MIN_SIZE = (30, 30)  # Minimum face size

# Visualization
FACE_RECT_COLOR = (0, 255, 0)  # Green
FACE_RECT_THICKNESS = 2
FACE_CENTER_COLOR = (0, 0, 255)  # Red
FACE_CENTER_RADIUS = 5
FONT = 'FONT_HERSHEY_SIMPLEX'
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 2

# Eye detection - Reduced strictness
EYE_DETECTION_ENABLED = True
EYE_MIN_NEIGHBORS = 2  # Reduced from 3
EYE_SCALE_FACTOR = 1.1
EYE_MIN_SIZE = (15, 15)  # Reduced minimum size

# Face geometry - Wider range
FACE_ASPECT_RATIO_MIN = 0.5  # Reduced minimum
FACE_ASPECT_RATIO_MAX = 1.1  # Increased maximum
MINIMUM_CONFIDENCE = 0.4  # Reduced confidence threshold

# Temporal filtering - Less strict
TEMPORAL_FILTERING_ENABLED = True
TEMPORAL_FRAMES_HISTORY = 5  # Frame history count
TEMPORAL_CONSISTENCY_THRESHOLD = 1  # Reduced from 3

# Motion detection - Less strict
MOTION_DETECTION_ENABLED = False  # Initially disabled for testing
MOTION_THRESHOLD = 5.0  # Reduced threshold

# Performance
MAX_FACES = 10  # Max faces to track
FRAME_SKIP = 2  # Process every n-th frame

# Debug options
DEBUG_MODE = True  # Enable debug outputs
