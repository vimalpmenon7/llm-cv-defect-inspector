"""Synthetic test images with known defects, used for tests and evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def make_surface(
    n_scratches: int = 2, n_spots: int = 1, size: int = 256, seed: int = 0
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Return (BGR image, ground-truth boxes) of a noisy gray surface with dark defects."""
    rng = np.random.default_rng(seed)
    base = rng.normal(150, 6, (size, size)).clip(0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    boxes: list[tuple[int, int, int, int]] = []

    for _ in range(n_scratches):
        x = int(rng.integers(20, size - 100))
        y = int(rng.integers(20, size - 20))
        length = int(rng.integers(50, 80))
        cv2.line(img, (x, y), (x + length, y + int(rng.integers(-3, 4))), (60, 60, 60), 2)
        boxes.append((x, y - 4, length, 8))

    for _ in range(n_spots):
        cx = int(rng.integers(30, size - 30))
        cy = int(rng.integers(30, size - 30))
        r = int(rng.integers(8, 14))
        cv2.circle(img, (cx, cy), r, (50, 50, 50), -1)
        boxes.append((cx - r, cy - r, 2 * r, 2 * r))
    return img, boxes
