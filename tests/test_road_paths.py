"""Tests for automatic road-path detection."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.road.path_estimator import RoadPathEstimator


def _five_path_frame() -> np.ndarray:
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    bottom_positions = [80, 210, 340, 470, 600, 730]
    top_positions = [300, 345, 390, 435, 480, 525]
    for bottom_x, top_x in zip(bottom_positions, top_positions):
        cv2.line(frame, (bottom_x, 470), (top_x, 140), (255, 255, 255), 5)
    return frame


class TestRoadPathEstimator(unittest.TestCase):
    def test_detects_five_paths_from_six_boundaries(self):
        count, confidence = RoadPathEstimator().estimate_frame(_five_path_frame())
        self.assertEqual(count, 5)
        self.assertGreaterEqual(confidence, 0.8)

    def test_blank_scene_uses_conservative_fallback(self):
        count, confidence = RoadPathEstimator().estimate_frame(
            np.zeros((480, 800, 3), dtype=np.uint8)
        )
        self.assertEqual(count, 1)
        self.assertEqual(confidence, 0.0)

    def test_guardrails_cannot_inflate_count_above_five_paths(self):
        frame = _five_path_frame()
        cv2.line(frame, (15, 470), (250, 140), (255, 255, 255), 5)
        cv2.line(frame, (785, 470), (550, 140), (255, 255, 255), 5)
        count, _ = RoadPathEstimator().estimate_frame(frame)
        self.assertEqual(count, 5)

    def test_video_uses_multi_frame_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "five_paths.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (800, 480)
            )
            for _ in range(15):
                writer.write(_five_path_frame())
            writer.release()

            estimate = RoadPathEstimator(sample_count=6).estimate_video(str(video_path))
            self.assertEqual(estimate.path_count, 5)
            self.assertGreaterEqual(estimate.confidence, 0.8)
            self.assertGreater(estimate.sampled_frames, 1)


if __name__ == "__main__":
    unittest.main()
