"""
Headless Face Detection CLI
Author: Romil V. Shah

Run face detection on an image or video file with no camera or display required.

Examples:
    python src/cli_detect.py --image face.jpg --json
    python src/cli_detect.py --image face.jpg --out annotated.jpg
    python src/cli_detect.py --video clip.mp4 --json
"""
import argparse
import json
import os
import sys

# Allow the bare intra-package imports (`import config`) used across src/.
sys.path.insert(0, os.path.dirname(__file__))

import cv2

from face_detector import FaceDetector
from nms_utils import apply_nms


def _serialize(face: dict) -> dict:
    return {
        "rect": list(face["rect"]),
        "center": list(face["center"]),
        "confidence": round(float(face["confidence"]), 4),
    }


def detect_in_image(path: str, detector: FaceDetector = None, max_faces: int = 10):
    """Return (frame, faces) for a single image file."""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    detector = detector or FaceDetector()
    faces = apply_nms(detector.detect_faces(frame, max_faces=max_faces))
    return frame, faces


def detect_in_video(path: str, detector: FaceDetector = None, max_faces: int = 10):
    """Return per-frame face counts for a video file (headless)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    detector = detector or FaceDetector()
    per_frame = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            per_frame.append(len(apply_nms(detector.detect_faces(frame, max_faces=max_faces))))
    finally:
        cap.release()
    return per_frame


def annotate(frame, faces):
    """Draw bounding boxes + confidence onto a copy of the frame."""
    out = frame.copy()
    for f in faces:
        x, y, w, h = f["rect"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{f['confidence']:.2f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless face detection on an image or video.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to an image file")
    src.add_argument("--video", help="Path to a video file")
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--out", help="Write an annotated image to this path (image mode only)")
    parser.add_argument("--json", action="store_true", help="Print results as JSON to stdout")
    args = parser.parse_args(argv)

    detector = FaceDetector()

    if args.image:
        frame, faces = detect_in_image(args.image, detector, args.max_faces)
        if args.out:
            cv2.imwrite(args.out, annotate(frame, faces))
        result = {"image": args.image, "count": len(faces), "faces": [_serialize(f) for f in faces]}
    else:
        counts = detect_in_video(args.video, detector, args.max_faces)
        result = {"video": args.video, "frames": len(counts), "faces_per_frame": counts,
                  "total_detections": sum(counts)}

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Detected {result.get('count', result.get('total_detections'))} face(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
