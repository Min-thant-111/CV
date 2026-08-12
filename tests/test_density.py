"""
Unit tests for DensityMetrics model and DensityEngine calculation module.
"""

import unittest
from backend.models.tracking import TrackedObject, FrameTracks
from backend.models.density import DensityMetrics
from backend.density.density_engine import DensityEngine, DensityConfig
from backend.density.vehicle_count_estimator import estimate_visible_vehicles


class TestDensityEngine(unittest.TestCase):
    """Test suite for traffic density calculation and level classification (LOW, MEDIUM, HIGH)."""

    def setUp(self):
        # Default capacity = 10.0 PCU units, LOW < 35%, HIGH >= 70%
        self.engine = DensityEngine(
            config=DensityConfig(
                max_road_capacity_units=10.0,
                low_threshold_pct=35.0,
                high_threshold_pct=70.0,
            )
        )

    def test_density_metrics_to_dict(self):
        metrics = DensityMetrics(
            frame_index=1,
            timestamp=0.033,
            total_vehicle_count=2,
            class_counts={"car": 1, "motorcycle": 1},
            weighted_vehicle_units=1.5,
            capacity_units=10.0,
            density_percentage=15.0,
            density_level="LOW",
            active_track_ids=[101, 102],
        )
        m_dict = metrics.to_dict()
        self.assertEqual(m_dict["density_level"], "LOW")
        self.assertEqual(m_dict["density_percentage"], 15.0)
        self.assertEqual(m_dict["weighted_vehicle_units"], 1.5)

    def test_low_resolution_five_path_queue_is_calibrated(self):
        estimate = estimate_visible_vehicles(
            detected_count=27,
            class_counts={"car": 26, "bus": 1},
            frame_width=320,
            frame_height=240,
            road_path_count=5,
        )

        self.assertEqual(estimate.detected_count, 27)
        self.assertEqual(estimate.estimated_count, 65)
        self.assertEqual(estimate.correction_factor, 2.4)
        self.assertEqual(sum(estimate.class_counts.values()), 65)
        self.assertEqual(estimate.class_counts, {"car": 63, "bus": 2})

    def test_sparse_low_resolution_scene_gets_modest_recall_correction(self):
        estimate = estimate_visible_vehicles(
            detected_count=18,
            class_counts={"car": 18},
            frame_width=320,
            frame_height=240,
            road_path_count=5,
        )

        self.assertEqual(estimate.estimated_count, 22)
        self.assertEqual(estimate.correction_factor, 1.2)

    def test_full_resolution_sparse_scene_is_not_inflated(self):
        estimate = estimate_visible_vehicles(
            detected_count=22,
            class_counts={"car": 22},
            frame_width=640,
            frame_height=480,
            road_path_count=5,
        )

        self.assertEqual(estimate.estimated_count, 22)
        self.assertEqual(estimate.correction_factor, 1.0)

    def test_trained_heavy_level_enables_occlusion_correction(self):
        estimate = estimate_visible_vehicles(
            detected_count=20,
            class_counts={"car": 20},
            frame_width=320,
            frame_height=240,
            road_path_count=5,
            traffic_level="heavy",
        )

        self.assertEqual(estimate.estimated_count, 52)
        self.assertEqual(estimate.correction_factor, 2.6)

    def test_trained_light_level_prevents_count_based_overcorrection(self):
        estimate = estimate_visible_vehicles(
            detected_count=30,
            class_counts={"car": 30},
            frame_width=320,
            frame_height=240,
            road_path_count=5,
            traffic_level="light",
        )

        self.assertEqual(estimate.estimated_count, 36)
        self.assertEqual(estimate.correction_factor, 1.2)

    def test_low_density_case(self):
        """Test LOW traffic density case (1 Car + 1 Motorcycle = 1.5 PCU -> 15%)."""
        tracks = [
            TrackedObject(track_id=1, class_id=2, class_name="car", confidence=0.9, bbox=(10, 10, 50, 50)),
            TrackedObject(track_id=2, class_id=3, class_name="motorcycle", confidence=0.8, bbox=(60, 60, 90, 90)),
        ]
        ft = FrameTracks(frame_index=0, timestamp=0.0, tracks=tracks)
        res = self.engine.compute_density(ft)

        self.assertEqual(res.total_vehicle_count, 2)
        self.assertEqual(res.weighted_vehicle_units, 1.5)
        self.assertEqual(res.density_percentage, 15.0)
        self.assertEqual(res.density_level, "LOW")

    def test_medium_density_case(self):
        """Test MEDIUM traffic density case (3 Cars + 1 Bus = 5.5 PCU -> 55%)."""
        tracks = [
            TrackedObject(track_id=1, class_id=2, class_name="car", confidence=0.9, bbox=(0, 0, 10, 10)),
            TrackedObject(track_id=2, class_id=2, class_name="car", confidence=0.9, bbox=(10, 10, 20, 20)),
            TrackedObject(track_id=3, class_id=2, class_name="car", confidence=0.9, bbox=(20, 20, 30, 30)),
            TrackedObject(track_id=4, class_id=5, class_name="bus", confidence=0.95, bbox=(30, 30, 70, 70)),
        ]
        ft = FrameTracks(frame_index=1, timestamp=0.033, tracks=tracks)
        res = self.engine.compute_density(ft)

        self.assertEqual(res.total_vehicle_count, 4)
        self.assertEqual(res.weighted_vehicle_units, 5.5)
        self.assertEqual(res.density_percentage, 55.0)
        self.assertEqual(res.density_level, "MEDIUM")

    def test_high_density_case(self):
        """Demand above capacity remains visible instead of being capped at 100%."""
        tracks = [
            TrackedObject(track_id=1, class_id=2, class_name="car", confidence=0.9, bbox=(0, 0, 10, 10)),
            TrackedObject(track_id=2, class_id=2, class_name="car", confidence=0.9, bbox=(10, 10, 20, 20)),
            TrackedObject(track_id=3, class_id=2, class_name="car", confidence=0.9, bbox=(20, 20, 30, 30)),
            TrackedObject(track_id=4, class_id=2, class_name="car", confidence=0.9, bbox=(30, 30, 40, 40)),
            TrackedObject(track_id=5, class_id=5, class_name="bus", confidence=0.9, bbox=(40, 40, 80, 80)),
            TrackedObject(track_id=6, class_id=5, class_name="bus", confidence=0.9, bbox=(80, 80, 120, 120)),
            TrackedObject(track_id=7, class_id=7, class_name="truck", confidence=0.9, bbox=(120, 120, 180, 180)),
        ]
        ft = FrameTracks(frame_index=2, timestamp=0.066, tracks=tracks)
        res = self.engine.compute_density(ft)

        self.assertEqual(res.total_vehicle_count, 7)
        self.assertEqual(res.weighted_vehicle_units, 12.0)
        self.assertEqual(res.density_percentage, 120.0)
        self.assertEqual(res.density_level, "HIGH")

    def test_same_vehicle_count_has_lower_density_across_more_paths(self):
        tracks = [
            TrackedObject(i, 2, "car", 0.9, (0, 0, 10, 10))
            for i in range(1, 31)
        ]
        frame_tracks = FrameTracks(frame_index=0, timestamp=0.0, tracks=tracks)

        one_path = DensityEngine(DensityConfig(road_path_count=1)).compute_density(frame_tracks)
        three_paths = DensityEngine(DensityConfig(road_path_count=3)).compute_density(frame_tracks)

        self.assertEqual(one_path.density_percentage, 300.0)
        self.assertEqual(three_paths.density_percentage, 100.0)
        self.assertGreater(one_path.density_percentage, three_paths.density_percentage)
        self.assertEqual(three_paths.road_path_count, 3)
        self.assertEqual(three_paths.capacity_units, 30.0)

    def test_roi_polygon_filtering(self):
        """Test that objects outside the ROI polygon are ignored."""
        # ROI polygon: square from (0,0) to (100,100)
        roi_polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.engine.set_roi_polygon(roi_polygon)

        tracks = [
            # Inside ROI (center = 25, 25)
            TrackedObject(track_id=101, class_id=2, class_name="car", confidence=0.9, bbox=(10, 10, 40, 40)),
            # Outside ROI (center = 200, 200)
            TrackedObject(track_id=102, class_id=2, class_name="car", confidence=0.9, bbox=(150, 150, 250, 250)),
        ]
        ft = FrameTracks(frame_index=3, timestamp=0.1, tracks=tracks)
        res = self.engine.compute_density(ft)

        self.assertEqual(res.total_vehicle_count, 1)
        self.assertEqual(res.active_track_ids, [101])


if __name__ == "__main__":
    unittest.main()
