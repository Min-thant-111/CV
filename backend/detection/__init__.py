"""
Object detection module (YOLO interface).
"""

from backend.detection.yolo_detector import (
    YOLODetector,
    DEFAULT_TARGET_CLASSES,
    YOLODetectorError,
    ModelNotFoundError,
)

__all__ = [
    "YOLODetector",
    "DEFAULT_TARGET_CLASSES",
    "YOLODetectorError",
    "ModelNotFoundError",
]
