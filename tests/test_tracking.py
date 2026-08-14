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

    @patch("ultralytics.YOLO")
    def test_detection_memory_bridges_short_misses_without_duplicates(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        box = MagicMock()
        box.cls = [MagicMock(item=lambda: 2)]
        box.conf = [MagicMock(item=lambda: 0.9)]
        box.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 40.0, 40.0])]
        box.id = 7
        detected = MagicMock()
        detected.boxes = [box]
        missed = MagicMock()
        missed.boxes = []

        mock_model.track.return_value = [detected]
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector, detection_memory_frames=2)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        self.assertEqual(tracker.track_frame(frame, frame_index=0).count, 1)
        mock_model.track.return_value = [missed]
        self.assertEqual(tracker.track_frame(frame, frame_index=1).count, 1)
        self.assertEqual(tracker.track_frame(frame, frame_index=2).count, 1)
        self.assertEqual(tracker.track_frame(frame, frame_index=3).count, 0)

    @patch("ultralytics.YOLO")
    def test_high_recall_tiles_cover_frame_and_add_non_duplicate(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        tracked_box = MagicMock()
        tracked_box.cls = [MagicMock(item=lambda: 2)]
        tracked_box.conf = [MagicMock(item=lambda: 0.9)]
        tracked_box.xyxy = [MagicMock(tolist=lambda: [5.0, 5.0, 35.0, 35.0])]
        tracked_box.id = 10
        tracked_result = MagicMock()
        tracked_result.boxes = [tracked_box]
        mock_model.track.return_value = [tracked_result]

        tile_box = MagicMock()
        tile_box.cls = [MagicMock(item=lambda: 7)]
        tile_box.conf = [MagicMock(item=lambda: 0.4)]
        tile_box.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 30.0, 30.0])]
        tile_results = []
        for index in range(4):
            result = MagicMock()
            result.boxes = [tile_box] if index == 1 else []
            tile_results.append(result)
        mock_model.predict.return_value = tile_results

        detector = YOLODetector(model_path="yolov8n.pt", inference_size=960)
        tracker = VehicleTracker(detector=detector, high_recall=True)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = tracker.track_frame(frame)

        self.assertEqual(tracker._tile_bounds(100, 100), [
            (0, 0, 62, 62), (38, 0, 100, 62),
            (0, 38, 62, 100), (38, 38, 100, 100),
        ])
        self.assertEqual(tracks.count, 2)
        self.assertEqual(tracks.tracks[0].track_id, 10)
        self.assertGreaterEqual(tracks.tracks[1].track_id, 1_000_000)
        self.assertEqual(tracks.tracks[1].bbox, (48.0, 10.0, 68.0, 30.0))

        carried = tracker.track_frame(frame, frame_index=1)
        self.assertEqual(carried.count, 2)
        self.assertEqual(mock_model.predict.call_count, 1)

        tracker.track_frame(frame, frame_index=5)
        self.assertEqual(mock_model.predict.call_count, 2)

    @patch("ultralytics.YOLO")
    def test_empty_high_recall_scan_waits_for_interval(self, mock_yolo_cls):
        """An empty supplemental scan must not be repeated on every frame."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        mock_model.track.return_value = [MagicMock(boxes=[])]
        mock_model.predict.return_value = [MagicMock(boxes=[]) for _ in range(4)]

        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(
            detector=detector,
            high_recall=True,
            tile_interval_frames=5,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        tracker.track_frame(frame, frame_index=0)
        tracker.track_frame(frame, frame_index=1)
        tracker.track_frame(frame, frame_index=4)
        self.assertEqual(mock_model.predict.call_count, 1)

        tracker.track_frame(frame, frame_index=5)
        self.assertEqual(mock_model.predict.call_count, 2)

    @patch("ultralytics.YOLO")
    def test_truck_requires_three_consistent_observations(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        truck_box = MagicMock()
        truck_box.cls = [MagicMock(item=lambda: 7)]
        truck_box.conf = [MagicMock(item=lambda: 0.82)]
        truck_box.xyxy = [MagicMock(tolist=lambda: [20.0, 15.0, 90.0, 85.0])]
        truck_box.id = 55
        result = MagicMock()
        result.boxes = [truck_box]
        mock_model.track.return_value = [result]

        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(
            detector=detector,
            heavy_vehicle_min_observations=3,
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        first = tracker.track_frame(frame, frame_index=0)
        second = tracker.track_frame(frame, frame_index=1)
        third = tracker.track_frame(frame, frame_index=2)

        self.assertEqual(first.tracks[0].class_name, "car")
        self.assertEqual(second.tracks[0].class_name, "car")
        self.assertEqual(third.tracks[0].class_id, 7)
        self.assertEqual(third.tracks[0].class_name, "truck")

    @patch("ultralytics.YOLO")
    def test_far_field_zoom_adds_distant_vehicle_in_frame_coordinates(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        tracked_result = MagicMock()
        tracked_result.boxes = []
        mock_model.track.return_value = [tracked_result]

        ordinary_tiles = [MagicMock(boxes=[]) for _ in range(4)]
        distant_box = MagicMock()
        distant_box.cls = [MagicMock(item=lambda: 2)]
        distant_box.conf = [MagicMock(item=lambda: 0.14)]
        distant_box.xyxy = [MagicMock(tolist=lambda: [10.0, 5.0, 20.0, 15.0])]
        far_results = [MagicMock(boxes=[]), MagicMock(boxes=[distant_box]), MagicMock(boxes=[])]
        mock_model.predict.side_effect = [ordinary_tiles, far_results]

        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(
            detector=detector,
            high_recall=True,
            far_field_recall=True,
        )
        tracks = tracker.track_frame(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(tracker._far_field_bounds(200, 100), [
            (0, 10, 96, 64), (60, 10, 132, 75), (90, 6, 200, 46),
        ])
        self.assertEqual(tracks.count, 1)
        self.assertEqual(tracks.tracks[0].bbox, (70.0, 15.0, 80.0, 25.0))
        self.assertEqual(mock_model.predict.call_args_list[1].kwargs["imgsz"], 1280)
        self.assertEqual(mock_model.predict.call_args_list[1].kwargs["conf"], 0.05)

    @patch("ultralytics.YOLO")
    def test_single_zoomed_tile_truck_never_relabels_tracked_car(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector)
        full_frame_car = TrackedObject(
            10, 2, "car", 0.60, (10.0, 10.0, 90.0, 90.0)
        )
        zoomed_truck = TrackedObject(
            -1, 7, "truck", 0.55, (12.0, 12.0, 92.0, 92.0)
        )

        merged = tracker._merge_supplemental([full_frame_car], [zoomed_truck])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].class_id, 2)
        self.assertEqual(merged[0].class_name, "car")
        self.assertEqual(merged[0].confidence, 0.60)

    @patch("ultralytics.YOLO")
    def test_weak_tile_truck_does_not_override_strong_car(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector)
        full_frame_car = TrackedObject(
            10, 2, "car", 0.90, (10.0, 10.0, 90.0, 90.0)
        )
        weak_truck = TrackedObject(
            -1, 7, "truck", 0.35, (12.0, 12.0, 92.0, 92.0)
        )

        merged = tracker._merge_supplemental([full_frame_car], [weak_truck])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].class_name, "car")

    @patch("ultralytics.YOLO")
    def test_three_confirmed_truck_frames_correct_an_initial_car_label(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector)

        first = tracker._stabilize_classes(
            [TrackedObject(22, 2, "car", 0.85, (0, 0, 50, 50))], {22}, 0
        )
        second = tracker._stabilize_classes(
            [TrackedObject(22, 7, "truck", 0.70, (0, 0, 50, 50))], {22}, 1
        )
        third = tracker._stabilize_classes(
            [TrackedObject(22, 7, "truck", 0.72, (0, 0, 50, 50))], {22}, 2
        )
        fourth = tracker._stabilize_classes(
            [TrackedObject(22, 7, "truck", 0.74, (0, 0, 50, 50))], {22}, 3
        )

        self.assertEqual(first[0].class_name, "car")
        self.assertEqual(second[0].class_name, "car")
        self.assertEqual(third[0].class_name, "car")
        self.assertEqual(fourth[0].class_name, "truck")

    @patch("ultralytics.YOLO")
    def test_two_confirmed_bus_frames_correct_an_initial_car_label(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        detector = YOLODetector(model_path="yolov8n.pt")
        tracker = VehicleTracker(detector=detector, bus_min_observations=2)

        first = tracker._stabilize_classes(
            [TrackedObject(23, 2, "car", 0.80, (0, 0, 50, 50))], {23}, 0
        )
        second = tracker._stabilize_classes(
            [TrackedObject(23, 5, "bus", 0.65, (0, 0, 50, 50))], {23}, 1
        )
        third = tracker._stabilize_classes(
            [TrackedObject(23, 5, "bus", 0.68, (0, 0, 50, 50))], {23}, 2
        )

        self.assertEqual(first[0].class_name, "car")
        self.assertEqual(second[0].class_name, "car")
        self.assertEqual(third[0].class_name, "bus")

    @patch("ultralytics.YOLO")
    def test_near_tie_car_truck_duplicate_prefers_car(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        tracker = VehicleTracker(detector=YOLODetector(model_path="yolov8n.pt"))
        consolidated = tracker._deduplicate_class_overlaps([
            TrackedObject(1, 7, "truck", 0.33, (20, 20, 80, 80)),
            TrackedObject(2, 2, "car", 0.31, (20, 20, 80, 80)),
        ])
        self.assertEqual(len(consolidated), 1)
        self.assertEqual(consolidated[0].class_name, "car")

    @patch("ultralytics.YOLO")
    def test_camera_text_boxes_are_removed_but_road_vehicles_remain(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        tracker = VehicleTracker(detector=YOLODetector(model_path="yolov8n.pt"))
        objects = [
            TrackedObject(1, 2, "car", 0.4, (2, 3, 22, 35)),
            TrackedObject(2, 2, "car", 0.8, (250, 20, 275, 48)),
            TrackedObject(3, 7, "truck", 0.8, (80, 40, 180, 120)),
        ]
        filtered = tracker._filter_camera_overlay_artifacts(objects, 320, 240)
        self.assertEqual([item.track_id for item in filtered], [2, 3])

    @patch("ultralytics.YOLO")
    def test_large_tall_vehicle_is_immediately_refined_to_truck(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        tracker = VehicleTracker(detector=YOLODetector(model_path="yolov8n.pt"))
        objects = [
            TrackedObject(10, 2, "car", 0.65, (100, 20, 140, 100)),
            TrackedObject(11, 2, "car", 0.70, (20, 70, 50, 100)),
            TrackedObject(12, 2, "car", 0.75, (55, 68, 87, 100)),
            TrackedObject(13, 2, "car", 0.72, (150, 69, 181, 100)),
        ]

        confirmed = tracker._refine_large_trucks(objects, 320, 240)
        stabilized = tracker._stabilize_classes(
            objects, {10, 11, 12, 13}, 0, geometry_heavy_ids=confirmed
        )

        self.assertEqual(confirmed, {10})
        self.assertEqual(stabilized[0].class_name, "truck")
        self.assertTrue(all(item.class_name == "car" for item in stabilized[1:]))

        briefly_occluded = tracker._stabilize_classes(
            [TrackedObject(10, 2, "car", 0.55, (102, 24, 141, 100))],
            {10},
            1,
        )
        self.assertEqual(briefly_occluded[0].class_name, "truck")

    @patch("ultralytics.YOLO")
    def test_low_confidence_giant_box_is_not_promoted_to_truck(self, mock_yolo_cls):
        mock_yolo_cls.return_value = MagicMock()
        tracker = VehicleTracker(detector=YOLODetector(model_path="yolov8n.pt"))
        objects = [
            TrackedObject(20, 2, "car", 0.12, (80, 5, 135, 105)),
            TrackedObject(21, 2, "car", 0.70, (20, 70, 50, 100)),
            TrackedObject(22, 2, "car", 0.75, (55, 68, 87, 100)),
        ]

        confirmed = tracker._refine_large_trucks(objects, 320, 240)

        self.assertEqual(confirmed, set())
        self.assertEqual(objects[0].class_name, "car")


if __name__ == "__main__":
    unittest.main()
