"""inspector: computer-vision defect detection with LLM-written inspection reports."""

from .detector import ContourDetector, Detection
from .pipeline import InspectionResult, analyze_image

__all__ = ["ContourDetector", "Detection", "InspectionResult", "analyze_image"]
__version__ = "0.1.0"
