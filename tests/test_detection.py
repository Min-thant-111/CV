"""
Unit tests for Detection models and YOLODetector module using mocked inference.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.models.detection import Detection, FrameDetections
from backend.detection.yolo_detector import (
    YOLODetector,
    ModelNotFoundError,
    YOLODetectorError,
)


class TestDetectionModels(unittest.TestCase):
    """Test suite for Detection data structures."""

    def test_detection_to_dict(self):
        det = Detection(
            class_id=2, class_name="car", confidence=0.87654, bbox=(10.5, 20.3, 100.1, 200.7)
        )
        d_dict = det.to_dict()
        self.assertEqual(d_dict["class_id"], 2)
        self.assertEqual(d_dict["class_name"], "car")
        self.assertEqual(d_dict["confidence"], 0.8765)
        self.assertEqual(d_dict["bbox"], [10.5, 20.3, 100.1, 200.7])

    def test_frame_detections_aggregations(self):
        dets = [
            Detection(2, "car", 0.9, (0, 0, 10, 10)),
            Detection(2, "car", 0.85, (10, 10, 20, 20)),
            Detection(3, "motorcycle", 0.75, (20, 20, 30, 30)),
            Detection(7, "truck", 0.88, (30, 30, 40, 40)),
        ]
        fd = FrameDetections(frame_index=1, timestamp=0.033, detections=dets)

        self.assertEqual(fd.count, 4)
        counts = fd.get_class_counts()
        self.assertEqual(counts, {"car": 2, "motorcycle": 1, "truck": 1})


class TestYOLODetector(unittest.TestCase):
    """Test suite for YOLODetector logic using mocks."""

    def test_invalid_model_path_raises_error(self):
        """Test that invalid non-existent model path raises ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError):
            YOLODetector(model_path="invalid_path/custom_model.pt")

    @patch("ultralytics.YOLO")
    def test_yolo_detection_filtering(self, mock_yolo_cls):
        """Test that YOLO detector properly filters for target vehicle classes."""
        # Setup mock YOLO model instance
        mock_model_instance = MagicMock()
        mock_yolo_cls.return_value = mock_model_instance

        # Create mock detection boxes
        # Box 1: Car (COCO 2)
        box_car = MagicMock()
        box_car.cls = [MagicMock(item=lambda: 2)]
        box_car.conf = [MagicMock(item=lambda: 0.88)]
        box_car.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 110.0, 120.0])]

        # Box 2: Motorcycle (COCO 3)
        box_bike = MagicMock()
        box_bike.cls = [MagicMock(item=lambda: 3)]
        box_bike.conf = [MagicMock(item=lambda: 0.92)]
        box_bike.xyxy = [MagicMock(tolist=lambda: [150.0, 200.0, 210.0, 260.0])]

        # Box 3: Person (COCO 0) - Should be filtered out
        box_person = MagicMock()
        box_person.cls = [MagicMock(item=lambda: 0)]
        box_person.conf = [MagicMock(item=lambda: 0.95)]
        box_person.xyxy = [MagicMock(tolist=lambda: [300.0, 300.0, 350.0, 400.0])]

        mock_result = MagicMock()
        mock_result.boxes = [box_car, box_bike, box_person]
        mock_model_instance.predict.return_value = [mock_result]

        # Instantiate detector
        detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.35)

        # Process dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_dets = detector.detect(dummy_frame, frame_index=42, timestamp=1.4)

        # Assertions
        self.assertEqual(frame_dets.frame_index, 42)
        self.assertEqual(frame_dets.timestamp, 1.4)
        # Person should be filtered out, leaving 2 vehicles
        self.assertEqual(frame_dets.count, 2)
        self.assertEqual(frame_dets.detections[0].class_name, "car")
        self.assertEqual(frame_dets.detections[1].class_name, "motorcycle")
        self.assertEqual(frame_dets.get_class_counts(), {"car": 1, "motorcycle": 1})
        call_kwargs = mock_model_instance.predict.call_args.kwargs
        self.assertEqual(call_kwargs["imgsz"], 640)
        self.assertEqual(call_kwargs["iou"], 0.45)
        self.assertEqual(call_kwargs["classes"], [2, 3, 5, 7])

    @patch("ultralytics.YOLO")
    def test_invalid_frame_raises_value_error(self, mock_yolo_cls):
        """Test that passing an invalid or empty frame raises ValueError."""
        detector = YOLODetector(model_path="yolov8n.pt")
        with self.assertRaises(ValueError):
            detector.detect(None)

        with self.assertRaises(ValueError):
            detector.detect(np.array([]))


if __name__ == "__main__":
    unittest.main()
