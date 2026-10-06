"""Defect detectors.

``ContourDetector`` is a classical OpenCV detector (background subtraction + contours)
that works without model weights. ``YoloDetector`` is an optional wrapper around an
Ultralytics YOLOv8 model you train on your own labelled defect data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected defect. ``bbox`` is (x, y, w, h) in pixels."""

    label: str
    bbox: tuple[int, int, int, int]
    area_ratio: float
    confidence: float
    severity: str

    def to_dict(self) -> dict:
        return asdict(self)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


def severity_from_area(area_ratio: float) -> str:
    """Map the fraction of the image covered by a defect to a severity level."""
    if area_ratio >= 0.02:
        return "high"
    if area_ratio >= 0.005:
        return "medium"
    return "low"


class ContourDetector:
    """Finds dark anomalies against a smooth background.

    Steps: grayscale -> large median blur as background estimate -> absolute difference ->
    Otsu threshold -> morphological clean-up -> contours. Elongated blobs are labelled
    ``scratch``; compact blobs are labelled ``spot``. Confidence is the mean contrast of the
    region (0-1), not a learned probability.
    """

    def __init__(self, min_area: int = 40, blur_ksize: int = 31, min_contrast: int = 25) -> None:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize must be odd")
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.min_contrast = min_contrast

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            raise ValueError("Empty image")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        background = cv2.medianBlur(gray, self.blur_ksize)
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask[diff < self.min_contrast] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total = float(gray.shape[0] * gray.shape[1])
        detections: list[Detection] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) < self.min_area and w * h < self.min_area:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            label = "scratch" if aspect >= 3.0 else "spot"
            region = diff[y : y + h, x : x + w]
            confidence = round(float(min(region.mean() / 255.0 * 2, 1.0)), 3)
            area_ratio = round(w * h / total, 5)
            detections.append(
                Detection(label, (x, y, w, h), area_ratio, confidence, severity_from_area(area_ratio))
            )
        return sorted(detections, key=lambda d: d.area_ratio, reverse=True)


class YoloDetector:
    """Optional YOLOv8 backend (``pip install ultralytics``). Requires your own weights.

    Not exercised by this repo's tests because no trained weights are bundled.
    """

    def __init__(self, weights_path: str, conf: float = 0.25) -> None:
        from ultralytics import YOLO  # imported lazily so the base install stays light

        self.model = YOLO(weights_path)
        self.conf = conf

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(image, conf=self.conf, verbose=False)[0]
        total = float(image.shape[0] * image.shape[1])
        out: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            area_ratio = round(w * h / total, 5)
            out.append(
                Detection(
                    result.names[int(box.cls[0])],
                    (x1, y1, w, h),
                    area_ratio,
                    round(float(box.conf[0]), 3),
                    severity_from_area(area_ratio),
                )
            )
        return sorted(out, key=lambda d: d.area_ratio, reverse=True)
