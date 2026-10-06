"""Glue: image -> detections -> report."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import ContourDetector, Detection, Detector
from .report import write_report


@dataclass
class InspectionResult:
    detections: list[Detection]
    report: str
    report_source: str
    verdict: str  # "pass" | "review" | "fail"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "report": self.report,
            "report_source": self.report_source,
            "detections": [d.to_dict() for d in self.detections],
        }


def verdict_from(detections: list[Detection]) -> str:
    if any(d.severity == "high" for d in detections):
        return "fail"
    return "review" if detections else "pass"


def analyze_image(
    image: np.ndarray, detector: Detector | None = None, use_llm: bool = True
) -> InspectionResult:
    """Run detection and report generation on a BGR image array."""
    detections = (detector or ContourDetector()).detect(image)
    report, source = write_report(detections, use_llm=use_llm)
    return InspectionResult(detections, report, source, verdict_from(detections))


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes; raises ``ValueError`` if not a valid image."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img
