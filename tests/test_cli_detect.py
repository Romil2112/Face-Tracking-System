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
