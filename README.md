# Real-Time Face Tracking System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

**Robust Real-Time Face Detection and Tracking with Advanced Error Recovery**

This comprehensive face tracking system combines traditional computer vision techniques with deep learning to achieve robust real-time performance. The implementation emphasizes error resilience, hardware acceleration support, and adaptive resource management, making it suitable for deployment across diverse environments. Built for stability rather than cutting-edge performance, it prioritizes error recovery over maximizing detection accuracy.

## Contributors
- **Romil V. Shah** - Lead Developer ([LinkedIn](https://linkedin.com/in/romil2112))
- **Parshav A. Shah** - Assistant Developer ([GitHub](https://github.com/pshah0601) | [LinkedIn](https://www.linkedin.com/in/parshav-shah6102))



## Table of Contents
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Error Recovery System](#error-recovery-system)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### Adaptive Detection Architecture
- Primary detector: Lightweight ResNet-SSD (300x300 resolution) for balance between speed and accuracy
- Fallback mechanism: Haar Cascade with dynamic parameter tuning (scale factor adjusts between 1.1–1.3 based on frame processing time)
- Automatic model fallback with health monitoring
- Temporal consistency validation: 5-frame buffer to confirm face presence before updating tracking coordinates

### Hardware Optimization
- CUDA/OpenCL acceleration with automatic backend selection
- Dynamic resource management (1-5 frame skip)
- Memory-optimized processing pipeline

### Fault Tolerance
- Three-stage error recovery system:
  1. CUDA memory exhaustion → Automatic CPU fallback
  2. Camera timeout → Hardware reset via USB power cycle emulation
  3. Model corruption → Local cache restoration from embedded weights
- Circuit breakers for critical subsystems
- Graceful degradation under load

## Technical Architecture
### Core Components
├── haarcascade_frontalface_default.xml  
├── LICENSE  
├── README.md  
├── requirements.txt  
└── src/    
* ├── main.py                     # Main application entry point
* ├── face_detector.py           # Hybrid DNN + Haar Cascade detection
* ├── video_capture.py           # Camera interface with error recovery
* ├── temporal_filter.py         # 5-frame temporal consistency checks
* ├── error_handling.py          # Circuit breakers & recovery system
* ├── motion_utils.py            # Optical flow-based motion analysis
* ├── tracking_visualizer.py     # BBox/FPS visualization
* ├── nms_utils.py               # Non-Maximum Suppression
* ├── config.py                  # Centralized configuration
* ├── __init__.py
* └── models/
  * ├── deploy.prototxt                   # DNN architecture
  * └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained weights  

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processor | Intel i5 8th Gen | Intel i7 11th Gen/NVIDIA GPU |
| RAM       | 8GB     | 16GB        |
| Storage   | 500MB   | 1GB SSD     |
| OS        | Windows 10 | Ubuntu 22.04 |
| Camera    | 720p Webcam | 1080p USB3.0 |

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

## Quick Start

1. git clone https://github.com/yourusername/face-tracking-app.git
2. cd face-tracking-app
3. pip install -r requirements.txt
4. python src/main.py --width 800 --height 600

## Configuration
You can modify various parameters in the `config.py` file to adjust the application's behavior.

## Usage Guide
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

## Contributing

1. Fork the repository
2. Create feature branch:
git checkout -b feature/improvement
3. Commit changes following [Semantic Commit](https://www.conventionalcommits.org) guidelines
4. Submit pull request with:
- Unit tests
- Updated documentation
- Performance benchmarks

## License
This project is licensed under the terms of the [LICENSE](https://github.com/Romil2112/Face-Tracking-System/blob/main/LICENSE) file included in this repository.
