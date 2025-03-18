"""
Configuration settings for the face tracking application.
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Camera backend settings
CAMERA_BACKEND_DSHOW = True  # Use DirectShow backend (recommended for Windows)
CAMERA_BACKEND_MSMF = False   # Use MSMF backend (may cause issues on some systems)

# Video format settings
CAMERA_FOURCC = 'MJPG'  # 'MJPG' tends to work better than default on many webcams

# Face detection settings
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)

# Visualization settings
FACE_RECT_COLOR = (0, 255, 0)  # Green
FACE_RECT_THICKNESS = 2
FACE_CENTER_COLOR = (0, 0, 255)  # Red
FACE_CENTER_RADIUS = 5
FONT = 'FONT_HERSHEY_SIMPLEX'
FONT_SCALE = 0.7
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 2

# Face verification parameters
EYE_DETECTION_ENABLED = True
EYE_MIN_NEIGHBORS = 3
EYE_SCALE_FACTOR = 1.1
EYE_MIN_SIZE = (20, 20)

# Temporal filtering parameters
TEMPORAL_FILTERING_ENABLED = True
TEMPORAL_FRAMES_HISTORY = 5
TEMPORAL_CONSISTENCY_THRESHOLD = 3

# Additional filtering parameters
FACE_ASPECT_RATIO_MIN = 0.7
FACE_ASPECT_RATIO_MAX = 0.9
MINIMUM_CONFIDENCE = 0.6

# Security settings
MAX_FACES = 10  # Maximum number of faces to track (for performance)
FRAME_SKIP = 2  # Process every nth frame for performance
