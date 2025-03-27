# Face Tracking Application
This comprehensive face tracking system combines traditional computer vision techniques with deep learning to achieve robust real-time performance. The implementation emphasizes error resilience, hardware acceleration support, and adaptive resource management, making it suitable for deployment across diverse environments.

## Key Features
Dual Detection Architecture: Hybrid approach using DNN (ResNet-SSD) and Haar Cascade classifiers with automatic fallback
Temporal Filtering: Motion prediction and consistency tracking across frames (5-frame history)
Hardware Acceleration: CUDA/OpenCL support with automatic backend selection
Circuit Breaker Pattern: Stateful error recovery for camera, detection, and resource subsystems
Dynamic Resource Management: Automatic frame skipping and load reduction during resource exhaustion
Non-Maximum Suppression: Hybrid OpenCV/manual IOU filtering with 0.4 threshold
Visualization System: Bounding box annotation with confidence scores and motion vectors

## Configuration
You can modify various parameters in the `config.py` file to adjust the application's behavior.

## License
This project is licensed under the terms of the LICENSE file included in this repository.

## Contributors
- **Romil V. Shah** - Lead Developer ([LinkedIn](https://linkedin.com/in/romil2112))
- **Parshav A. Shah** - Assistant Developer ([GitHub](https://github.com/pshah0601) | [LinkedIn](https://www.linkedin.com/in/parshav-shah6102))

## Installation

Prerequisites
- Python 3.8+
- OpenCV 4.5+ with contrib modules
- CUDA Toolkit 11.0+ (optional)
- NVIDIA GPU with Compute Capability 3.0+ (for CUDA acceleration)

1. Clone this repository:
git clone https://github.com/yourusername/face-tracking-app.git
cd face-tracking-app

2. Install the required dependencies:
pip install -r requirements.txt

3. Ensure you have the `haarcascade_frontalface_default.xml` file in the root directory of the project.

## Usage
To run the application:

1. Navigate to the `src` folder:
cd src

2. Run the main script:
python main.py

Optional command-line arguments:
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--cascade`: Path to Haar cascade XML file
- `--scale-factor`: Scale factor for face detection
- `--min-neighbors`: Min neighbors for face detection
- `--max-faces`: Maximum number of faces to track
- `--debug`: Enable debug output

## Error Recovery System:
Three-tier fault tolerance mechanism:
1. Retry Decorator: Exponential backoff with jitter (3 attempts)
2. Circuit Breakers: State tracking for camera/detector subsystems
3. Graceful Degradation:
  - Dynamic frame skipping
  - Model complexity reduction
  - CUDA → CPU fallback
