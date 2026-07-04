"""Tests for the headless detection CLI helpers."""
import cv2
import numpy as np
import pytest

import cli_detect


def test_detect_in_image_returns_frame_and_list(tmp_path):
    img_path = tmp_path / "blank.png"
    cv2.imwrite(str(img_path), np.zeros((120, 120, 3), dtype=np.uint8))
    frame, faces = cli_detect.detect_in_image(str(img_path))
    assert frame is not None
    assert isinstance(faces, list)


def test_detect_in_image_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        cli_detect.detect_in_image("/nonexistent/path/to/image.png")


def test_annotate_preserves_frame_shape(tmp_path):
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    faces = [{"rect": (10, 10, 20, 20), "center": (20, 20), "confidence": 0.9}]
    out = cli_detect.annotate(frame, faces)
    assert out.shape == frame.shape
    # Annotation must not mutate the original frame in place.
    assert np.array_equal(frame, np.zeros((80, 80, 3), dtype=np.uint8))


def test_main_image_mode_json(tmp_path, capsys):
    img_path = tmp_path / "blank.png"
    cv2.imwrite(str(img_path), np.zeros((120, 120, 3), dtype=np.uint8))
    rc = cli_detect.main(["--image", str(img_path), "--json"])
    assert rc == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert out["count"] == len(out["faces"])


def test_serialize_rounds_and_lists():
    face = {"rect": (1, 2, 3, 4), "center": (5, 6), "confidence": 0.987654}
    s = cli_detect._serialize(face)
    assert s["rect"] == [1, 2, 3, 4]
    assert s["center"] == [5, 6]
    assert s["confidence"] == 0.9877


def _write_tiny_video(path, n_frames=2, size=(64, 64)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    assert writer.isOpened(), "could not open VideoWriter"
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_detect_in_video_returns_per_frame_counts(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_tiny_video(vid, n_frames=3)
    counts = cli_detect.detect_in_video(str(vid))
    assert isinstance(counts, list)
    assert all(isinstance(c, int) for c in counts)


def test_detect_in_video_raises_on_bad_file():
    with pytest.raises(FileNotFoundError):
        cli_detect.detect_in_video("/nonexistent/clip.mp4")


def test_main_image_mode_with_out(tmp_path, capsys):
    img_path = tmp_path / "blank.png"
    out_path = tmp_path / "annotated.png"
    cv2.imwrite(str(img_path), np.zeros((120, 120, 3), dtype=np.uint8))
    rc = cli_detect.main(["--image", str(img_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    # Non-JSON mode prints a human-readable summary.
    assert "face(s)" in capsys.readouterr().out


def test_main_video_mode_json(tmp_path, capsys):
    vid = tmp_path / "clip.mp4"
    _write_tiny_video(vid, n_frames=2)
    rc = cli_detect.main(["--video", str(vid), "--json"])
    assert rc == 0
    import json
    out = json.loads(capsys.readouterr().out)
    assert "total_detections" in out
    assert out["frames"] == len(out["faces_per_frame"])
