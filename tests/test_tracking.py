"""
Unit tests for TrackedObject, FrameTracks models, and VehicleTracker module using mocked tracking results.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.models.tracking import TrackedObject, FrameTracks
from backend.tracking.tracker import VehicleTracker
from backend.detection.yolo_detector import YOLODetector


class TestTrackingModels(unittest.TestCase):
    """Test suite for Tracking data structures."""

    def test_tracked_object_to_dict(self):
        obj = TrackedObject(
            track_id=101,
            class_id=2,
            class_name="car",
            confidence=0.9123,
            bbox=(50.0, 100.0, 200.0, 300.0),
        )
        d_dict = obj.to_dict()
        self.assertEqual(d_dict["track_id"], 101)
        self.assertEqual(d_dict["class_name"], "car")
        self.assertEqual(d_dict["confidence"], 0.9123)
        self.assertEqual(d_dict["bbox"], [50.0, 100.0, 200.0, 300.0])

    def test_frame_tracks_helpers(self):
        tracks = [
            TrackedObject(1, 2, "car", 0.9, (0, 0, 10, 10)),
            TrackedObject(2, 2, "car", 0.85, (10, 10, 20, 20)),
            TrackedObject(3, 5, "bus", 0.95, (20, 20, 30, 30)),
        ]
        ft = FrameTracks(frame_index=10, timestamp=0.33, tracks=tracks)

        self.assertEqual(ft.count, 3)
        self.assertEqual(ft.get_track_ids(), [1, 2, 3])
        self.assertEqual(ft.get_class_counts(), {"car": 2, "bus": 1})


class TestVehicleTracker(unittest.TestCase):
    """Test suite for VehicleTracker using mocked ByteTrack predictions."""

    @patch("ultralytics.YOLO")
    def test_tracking_persistence_and_filtering(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock Frame 1 detection with track IDs 101 (Car) and 102 (Motorcycle)
        box1 = MagicMock()
        box1.cls = [MagicMock(item=lambda: 2)]  # car
        box1.conf = [MagicMock(item=lambda: 0.90)]
        box1.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 150.0])]
        box1.id = 101

        box2 = MagicMock()
        box2.cls = [MagicMock(item=lambda: 3)]  # motorcycle
        box2.conf = [MagicMock(item=lambda: 0.85)]
        box2.xyxy = [MagicMock(tolist=lambda: [200.0, 250.0, 250.0, 300.0])]
        box2.id = 102

        mock_res1 = MagicMock()
        mock_res1.boxes = [box1, box2]

        mock_model.track.return_value = [mock_res1]

        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1_tracks = tracker.track_frame(dummy_frame, frame_index=0, timestamp=0.0)

        self.assertEqual(frame1_tracks.count, 2)
        self.assertEqual(frame1_tracks.get_track_ids(), [101, 102])
        self.assertEqual(frame1_tracks.tracks[0].track_id, 101)
        self.assertEqual(frame1_tracks.tracks[0].class_name, "car")

        # Mock Frame 2 where Car 101 persists, Motorcycle exits, Bus 103 enters
        box3_bus = MagicMock()
        box3_bus.cls = [MagicMock(item=lambda: 5)]  # bus
        box3_bus.conf = [MagicMock(item=lambda: 0.94)]
        box3_bus.xyxy = [MagicMock(tolist=lambda: [300.0, 100.0, 500.0, 350.0])]
        box3_bus.id = 103

        mock_res2 = MagicMock()
        mock_res2.boxes = [box1, box3_bus]

        mock_model.track.return_value = [mock_res2]

        frame2_tracks = tracker.track_frame(dummy_frame, frame_index=1, timestamp=0.033)

        self.assertEqual(frame2_tracks.count, 2)
        self.assertEqual(frame2_tracks.get_track_ids(), [101, 103])
        self.assertEqual(frame2_tracks.tracks[0].track_id, 101)  # Persistent ID 101
        self.assertEqual(frame2_tracks.tracks[1].track_id, 103)  # New ID 103

    @patch("ultralytics.YOLO")
    def test_invalid_frame_raises_error(self, mock_yolo_cls):
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector)

        with self.assertRaises(ValueError):
            tracker.track_frame(None)


if __name__ == "__main__":
    unittest.main()
