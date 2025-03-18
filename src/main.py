"""
Main module for the face tracking application.
"""

import cv2
import time
import argparse
from typing import Dict, Any

from face_detector import FaceDetector
from video_capture import VideoCapture
from tracking_visualizer import TrackingVisualizer
import config

def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary containing the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Real-time face tracking using OpenCV')
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX,
                        help=f'Camera index (default: {config.CAMERA_INDEX})')
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH,
                        help=f'Camera width (default: {config.CAMERA_WIDTH})')
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT,
                        help=f'Camera height (default: {config.CAMERA_HEIGHT})')
    parser.add_argument('--cascade', type=str, default=config.CASCADE_PATH,
                        help=f'Path to Haar cascade XML file (default: {config.CASCADE_PATH})')
    parser.add_argument('--scale-factor', type=float, default=config.SCALE_FACTOR,
                        help=f'Scale factor for face detection (default: {config.SCALE_FACTOR})')
    parser.add_argument('--min-neighbors', type=int, default=config.MIN_NEIGHBORS,
                        help=f'Min neighbors for face detection (default: {config.MIN_NEIGHBORS})')
    parser.add_argument('--max-faces', type=int, default=config.MAX_FACES,
                        help=f'Maximum number of faces to track (default: {config.MAX_FACES})')
    
    return vars(parser.parse_args())

def main():
    """
    Main function to run the face tracking application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize face detector
    try:
        face_detector = FaceDetector(
            cascade_path=args['cascade'],
            scale_factor=args['scale_factor'],
            min_neighbors=args['min_neighbors'],
            min_size=config.MIN_SIZE
        )
        print(f"Face detector initialized with cascade: {args['cascade']}")
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        return
    
    # Initialize video capture
    video_capture = VideoCapture(
        camera_index=args['camera'],
        width=args['width'],
        height=args['height'],
        fps=config.CAMERA_FPS
    )
    
    if not video_capture.start():
        print("Error starting video capture")
        return
    
    print(f"Video capture started with camera index: {args['camera']}")
    
    # Initialize tracking visualizer
    visualizer = TrackingVisualizer(
        rect_color=config.FACE_RECT_COLOR,
        rect_thickness=config.FACE_RECT_THICKNESS,
        center_color=config.FACE_CENTER_COLOR,
        center_radius=config.FACE_CENTER_RADIUS,
        font=getattr(cv2, config.FONT),
        font_scale=config.FONT_SCALE,
        font_color=config.FONT_COLOR,
        font_thickness=config.FONT_THICKNESS
    )
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Frame processing variables
    process_this_frame = 0
    faces = []
    
    print("Face tracking started. Press 'q' to quit.")
    
    # Main loop
    while True:
        # Read a frame from the video capture
        success, frame = video_capture.read()
        
        if not success:
            print("Error reading frame from video capture")
            break
        
        # Process every nth frame for better performance
        if process_this_frame == 0:
            # Detect faces
            faces = face_detector.detect_faces(frame, max_faces=args['max_faces'])
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Update FPS in visualizer
            visualizer.set_fps(fps)
        
        # Visualize the results
        output_frame = visualizer.draw_faces(frame, faces)
        
        # Display the output
        cv2.imshow('Face Tracking', output_frame)
        
        # Update frame processing counter
        process_this_frame = (process_this_frame + 1) % config.FRAME_SKIP
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up
    video_capture.stop()
    cv2.destroyAllWindows()
    print("Face tracking stopped")

if __name__ == '__main__':
    main()
