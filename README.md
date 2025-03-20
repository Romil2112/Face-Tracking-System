# Face Tracking Application

## Features
- Real-time face detection and tracking
- Eye detection for improved face verification
- Temporal filtering to reduce false positives
- Motion detection capabilities
- Customizable visualization

## Configuration
You can modify various parameters in the `config.py` file to adjust the application's behavior.

## License
This project is licensed under the terms of the LICENSE file included in this repository.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

This application performs real-time face tracking using OpenCV and Haar Cascades.

## Installation
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

## Executable Version

For users who prefer not to set up the Python environment, we provide a pre-compiled executable version of the Face Tracking Application.

### Using the Executable

1. Download the `FaceTrackingApp.exe` file from the repository.
2. Ensure you have the `haarcascade_frontalface_default.xml` file in the same directory as the executable.
3. Double-click on `FaceTrackingApp.exe` to run the application.

Note: The executable version may have slightly slower performance compared to running the Python script directly. It also may trigger warnings from some antivirus software due to its nature as an unsigned executable.

If you encounter any issues with the executable, please try running the Python script version as described in the Installation and Usage sections above.
