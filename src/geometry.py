"""Geometry helpers shared across the tracking pipeline."""
import logging

logger = logging.getLogger(__name__)

__all__ = ["calculate_iou"]


def calculate_iou(box1, box2) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes.

    Returns 0.0 for non-overlapping or degenerate (zero-area) boxes, and on any
    malformed input rather than raising — callers use this inside detection loops.
    """
    try:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    except (ValueError, TypeError) as e:
        logger.error(f"IoU calculation error: {e}")
        return 0.0
