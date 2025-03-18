"""
Configuration settings for the face tracking application.
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

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

# Security settings
MAX_FACES = 10  # Maximum number of faces to track (for performance)
FRAME_SKIP = 2  # Process every nth frame for performance
