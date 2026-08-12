"""
YOLO Vehicle Detector module wrapping Ultralytics YOLO for single-frame inference.
"""

from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

from backend.models.detection import Detection, FrameDetections


# Default traffic-related COCO vehicle class mappings:
# 2: car, 3: motorcycle, 5: bus, 7: truck
DEFAULT_TARGET_CLASSES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLODetectorError(Exception):
    """Base exception for YOLO detector errors."""

    pass


class ModelNotFoundError(YOLODetectorError, FileNotFoundError):
    """Raised when the specified YOLO weights file cannot be found."""

    pass


class YOLODetector:
    """Ultralytics YOLO wrapper for real-time traffic vehicle detection."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.35,
        target_classes: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        inference_size: int = 640,
        iou_threshold: float = 0.45,
    ):
        """Args:

        model_path: Path or identifier for YOLO weights (.pt, .onnx, .engine).
        confidence_threshold: Minimum confidence score (0.0 to 1.0) to filter detections.
        target_classes: Dict mapping class_id -> class_name for filtering.
        device: Computing device ('cpu', '0', 'cuda', etc.).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or DEFAULT_TARGET_CLASSES
        self.device = device
        self.inference_size = max(320, int(inference_size))
        self.iou_threshold = min(1.0, max(0.05, float(iou_threshold)))
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Ultralytics YOLO model from specified path with validation."""
        path_obj = Path(self.model_path)
        # Validate custom local paths if file does not exist and isn't standard ultralytics weight name
        if not path_obj.exists() and ("/" in self.model_path or "\\" in self.model_path):
            raise ModelNotFoundError(
                f"Specified YOLO model path does not exist: '{self.model_path}'"
            )

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
        except Exception as e:
            raise YOLODetectorError(
                f"Failed to load YOLO model from '{self.model_path}': {e}"
            ) from e

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> FrameDetections:
        """Run vehicle detection on a single OpenCV BGR frame.

        Args:
            frame: OpenCV BGR image (np.ndarray).
            frame_index: Index of current frame in video stream.
            timestamp: Timestamp in seconds.

        Returns:
            FrameDetections object containing filtered vehicle detections.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid input frame provided for YOLO detection.")

        if self.model is None:
            raise YOLODetectorError("YOLO model is not initialized.")

        # Run inference on single frame
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.inference_size,
            classes=list(self.target_classes),
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    # Filter for target traffic classes only
                    if cls_id in self.target_classes:
                        class_name = self.target_classes[cls_id]
                        detections.append(
                            Detection(
                                class_id=cls_id,
                                class_name=class_name,
                                confidence=round(conf, 4),
                                bbox=(
                                    round(xyxy[0], 2),
                                    round(xyxy[1], 2),
                                    round(xyxy[2], 2),
                                    round(xyxy[3], 2),
                                ),
                            )
                        )

        return FrameDetections(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=detections,
        )
