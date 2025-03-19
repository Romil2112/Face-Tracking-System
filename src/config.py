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

# Face detection
CASCADE_PATH = 'haarcascade_frontalface_default.xml'  # Haar cascade file
SCALE_FACTOR = 1.1  # Image scaling factor
MIN_NEIGHBORS = 5  # Detection quality
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

# Eye detection
EYE_DETECTION_ENABLED = True
EYE_MIN_NEIGHBORS = 3
EYE_SCALE_FACTOR = 1.1
EYE_MIN_SIZE = (20, 20)

# Face geometry
FACE_ASPECT_RATIO_MIN = 0.7  # Min width/height ratio
FACE_ASPECT_RATIO_MAX = 0.9  # Max width/height ratio
MINIMUM_CONFIDENCE = 0.6  # Min detection confidence

# Temporal filtering
TEMPORAL_FILTERING_ENABLED = True
TEMPORAL_FRAMES_HISTORY = 5  # Frame history count
TEMPORAL_CONSISTENCY_THRESHOLD = 3  # Min frame consistency

# Motion detection
MOTION_DETECTION_ENABLED = True
MOTION_THRESHOLD = 10.0  # Min motion magnitude

# Performance
MAX_FACES = 10  # Max faces to track
FRAME_SKIP = 2  # Process every n-th frame
