"""
Face Tracking Application - Main Module
Author: Romil V. Shah
This module contains the main execution logic for the face tracking application.
"""

import cv2
import time
import argparse
from typing import Dict, Any
import numpy as np
from face_detector import FaceDetector
from video_capture import VideoCapture
from tracking_visualizer import TrackingVisualizer
import config
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def parse_arguments() -> Dict[str, Any]:
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
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    return vars(parser.parse_args())

def main():
    args = parse_arguments()
    if args['debug']:
        config.DEBUG_MODE = True

    try:
        face_detector = FaceDetector(
            cascade_path=args['cascade'],
            scale_factor=args['scale_factor'],
            min_neighbors=args['min_neighbors'],
            min_size=config.MIN_SIZE
        )
        print(f"Face detector initialized successfully")
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        print("Please ensure the Haar cascade XML file exists and is valid")
        print("You can download it from: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        return

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

    temporal_filter = None
    if config.TEMPORAL_FILTERING_ENABLED:
        try:
            from temporal_filter import TemporalFilter
            temporal_filter = TemporalFilter(
                history_size=config.TEMPORAL_FRAMES_HISTORY,
                consistency_threshold=config.TEMPORAL_CONSISTENCY_THRESHOLD
            )
            print("Temporal filtering enabled")
        except ImportError as e:
            print(f"Warning: Could not import TemporalFilter: {e}")
            print("Temporal filtering will be disabled")
        except Exception as e:
            print(f"Warning: Error initializing temporal filter: {e}")
            print("Temporal filtering will be disabled")

    motion_detection = config.MOTION_DETECTION_ENABLED
    if motion_detection:
        try:
            from motion_utils import detect_motion
            print("Motion detection enabled")
        except ImportError as e:
            print(f"Warning: Could not import motion_utils: {e}")
            motion_detection = False

    prev_frame = None
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

    frame_count = 0
    start_time = time.time()
    fps = 0
    process_this_frame = 0
    faces = []

    print("Face tracking started. Press 'q' to quit.")
    cv2.namedWindow('Face Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Tracking', config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    retry_count = 0
    max_retries = 3
    while True:
        success, frame = video_capture.read()
        if not success:
            retry_count += 1
            print(f"Error reading frame from video capture (Attempt {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print("Maximum retry attempts reached. Stopping face tracking.")
                break
            time.sleep(1)
            continue

        retry_count = 0
        frame = cv2.flip(frame, 1)  # Horizontal flip

        if process_this_frame == 0:
            faces = face_detector.detect_faces(frame, max_faces=args['max_faces'])
            if config.DEBUG_MODE:
                print(f"Detected {len(faces)} faces")
                for i, face in enumerate(faces):
                    print(f" Face #{i+1}: confidence={face.get('confidence', 'N/A')}, eyes={face.get('eye_count', 'N/A')}")

            if motion_detection and prev_frame is not None:
                try:
                    motion_magnitude = detect_motion(prev_frame, frame)
                    if motion_magnitude is not None:
                        motion_faces = []
                        for face in faces:
                            x, y, w, h = face['rect']
                            face_motion = np.mean(motion_magnitude[y:y+h, x:x+w])
                            face['motion'] = face_motion
                            if face_motion >= config.MOTION_THRESHOLD:
                                motion_faces.append(face)
                            elif config.DEBUG_MODE:
                                print(f"Face rejected by motion: {face_motion} < {config.MOTION_THRESHOLD}")
                        if config.DEBUG_MODE:
                            print(f"Motion filtering: {len(faces)} → {len(motion_faces)}")
                        faces = motion_faces
                except Exception as e:
                    if config.DEBUG_MODE:
                        print(f"Error during motion detection: {e}")

            if motion_detection:
                prev_frame = frame.copy()

            if temporal_filter is not None:
                try:
                    faces = temporal_filter.update(faces)
                except Exception as e:
                    if config.DEBUG_MODE:
                        print(f"Error during temporal filtering: {e}")

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        visualizer.set_fps(fps)
        output_frame = visualizer.draw_faces(frame, faces)

        cv2.imshow('Face Tracking', output_frame)

        process_this_frame = (process_this_frame + 1) % config.FRAME_SKIP

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_VISIBLE) < 1:
            break

    video_capture.stop()
    cv2.destroyAllWindows()
    print("Face tracking stopped")

if __name__ == '__main__':
    main()
