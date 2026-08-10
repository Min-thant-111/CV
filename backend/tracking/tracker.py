"""
Vehicle Tracker module leveraging Ultralytics ByteTrack for multi-object tracking.
"""

from typing import List, Optional, Dict
import numpy as np

from backend.detection.yolo_detector import YOLODetector
from backend.models.tracking import TrackedObject, FrameTracks


class TrackerError(Exception):
    """Base exception for tracker errors."""

    pass


class VehicleTracker:
    """ByteTrack wrapper integrating with Ultralytics YOLO for persistent vehicle tracking."""

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.35,
        tracker_type: str = "bytetrack.yaml",
        target_classes: Optional[Dict[int, str]] = None,
    ):
        """Args:

        detector: Existing YOLODetector instance, or None to create a new one.
        model_path: Path to YOLO model weights if detector is None.
        confidence_threshold: Confidence score threshold.
        tracker_type: Ultralytics tracker configuration ('bytetrack.yaml' or 'botsort.yaml').
        target_classes: Dict mapping class_id -> class_name for vehicle filtering.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLODetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                target_classes=target_classes,
            )

        self.tracker_type = tracker_type
        self.target_classes = target_classes or self.detector.target_classes

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> FrameTracks:
        """Process a single frame sequentially and return persistent object tracks.

        Args:
            frame: OpenCV BGR image matrix.
            frame_index: Frame sequence index.
            timestamp: Frame timestamp in seconds.

        Returns:
            FrameTracks dataclass containing list of TrackedObject instances.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame provided for vehicle tracking.")

        if self.detector.model is None:
            raise TrackerError("Underlying YOLO model is not initialized.")

        # Run ByteTrack multi-object tracking via Ultralytics
        results = self.detector.model.track(
            source=frame,
            conf=self.detector.confidence_threshold,
            persist=True,  # Maintains track states across consecutive frames
            tracker=self.tracker_type,
            device=self.detector.device,
            verbose=False,
        )

        tracks: List[TrackedObject] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                    # Filter for target vehicle classes
                    if cls_id in self.target_classes:
                        class_name = self.target_classes[cls_id]

                        # Extract track ID safely from box.id or fallback to (idx + 1)
                        track_id = idx + 1
                        if hasattr(box, "id") and box.id is not None:
                            val = box.id
                            if hasattr(val, "item"):
                                track_id = int(val.item())
                            elif hasattr(val, "__getitem__") and len(val) > 0:
                                elem = val[0]
                                track_id = int(elem.item()) if hasattr(elem, "item") else int(elem)
                            else:
                                try:
                                    track_id = int(val)
                                except (ValueError, TypeError):
                                    pass

                        tracks.append(
                            TrackedObject(
                                track_id=track_id,
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

        return FrameTracks(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=tracks,
        )
