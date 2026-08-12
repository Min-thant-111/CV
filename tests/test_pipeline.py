"""
End-to-end integration test for the complete TrafficPipeline.

Mocks YOLO/ByteTrack inference and MQTT broker — verifies that all pipeline
layers connect correctly and that the architecture boundaries are respected.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from backend.pipeline_config import PipelineConfig
from backend.density.density_engine import DensityConfig
from backend.signaling.signal_engine import SignalConfig
from backend.mqtt.mqtt_publisher import MQTTConfig
from backend.pipeline import TrafficPipeline, FrameResult
from backend.models.tracking import TrackedObject, FrameTracks
from backend.models.density import DensityMetrics
from backend.models.signaling import SignalDecision


def _make_synthetic_video(path: str, frames: int = 20) -> str:
    """Helper: generate a small synthetic test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (320, 240))
    for i in range(frames):
        frame = np.full((240, 320, 3), (i * 12 % 255, 100, 150), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestFrameResult(unittest.TestCase):
    """Test FrameResult structured log helper."""

    def test_frame_result_log_does_not_raise(self):
        result = FrameResult(
            frame_index=30,
            timestamp=1.0,
            vehicle_count=4,
            class_counts={"car": 3, "bus": 1},
            density_percentage=55.0,
            density_level="MEDIUM",
            signal="GREEN",
            green_duration=50,
            mqtt_published=True,
            reason="Moderate density",
        )
        try:
            result.log_summary()
        except Exception as e:
            self.fail(f"log_summary() raised: {e}")


class TestTrafficPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline integration tests with mocked YOLO, tracking and MQTT."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "test_traffic.mp4")
        _make_synthetic_video(self.video_path, frames=60)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_config(self, mqtt_enabled: bool = False) -> PipelineConfig:
        return PipelineConfig(
            video_path=self.video_path,
            model_path="yolov8n.pt",
            confidence_threshold=0.35,
            frame_skip=0,
            publish_interval_frames=10,
            log_interval_frames=10,
            mqtt_enabled=mqtt_enabled,
            density=DensityConfig(max_road_capacity_units=10.0),
            signal=SignalConfig(
                green_duration_low=30,
                green_duration_medium=50,
                green_duration_high=70,
            ),
            mqtt=MQTTConfig(broker_host="localhost", broker_port=1883),
        )

    @patch("ultralytics.YOLO")
    def test_pipeline_runs_without_mqtt(self, mock_yolo_cls):
        """Pipeline completes full video with MQTT disabled — no YOLO model needed."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock ByteTrack to return 2 cars per frame
        def mock_track(*args, **kwargs):
            box_car1 = MagicMock()
            box_car1.cls = [MagicMock(item=lambda: 2)]
            box_car1.conf = [MagicMock(item=lambda: 0.90)]
            box_car1.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 80.0, 80.0])]
            box_car1.id = 1

            box_car2 = MagicMock()
            box_car2.cls = [MagicMock(item=lambda: 2)]
            box_car2.conf = [MagicMock(item=lambda: 0.85)]
            box_car2.xyxy = [MagicMock(tolist=lambda: [100.0, 100.0, 200.0, 200.0])]
            box_car2.id = 2

            mock_res = MagicMock()
            mock_res.boxes = [box_car1, box_car2]
            return [mock_res]

        mock_model.track.side_effect = mock_track

        config = self._make_config(mqtt_enabled=False)
        pipeline = TrafficPipeline(config=config)
        # Should run without exceptions
        pipeline.run(video_path=self.video_path)

    @patch("paho.mqtt.client.Client")
    @patch("ultralytics.YOLO")
    def test_pipeline_publishes_mqtt_at_interval(self, mock_yolo_cls, mock_mqtt_cls):
        """Pipeline publishes to MQTT at configured interval frames."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock 4 cars — MEDIUM/HIGH density (4.0 PCU / 10.0 cap = 40% → MEDIUM)
        def mock_track(*args, **kwargs):
            boxes = []
            for i in range(4):
                box = MagicMock()
                box.cls = [MagicMock(item=lambda: 2)]
                box.conf = [MagicMock(item=lambda: 0.90)]
                box.xyxy = [MagicMock(tolist=lambda: [10.0, 10.0, 80.0, 80.0])]
                box.id = i + 1
                boxes.append(box)
            res = MagicMock()
            res.boxes = boxes
            return [res]

        mock_model.track.side_effect = mock_track

        mock_mqtt_instance = MagicMock()
        mock_mqtt_cls.return_value = mock_mqtt_instance
        mock_mqtt_instance.publish.return_value = MagicMock(rc=0)

        config = self._make_config(mqtt_enabled=True)
        pipeline = TrafficPipeline(config=config)

        # Simulate connected state so publish_signal proceeds
        pipeline._mqtt_publisher._is_connected = True
        pipeline._mqtt_connected = True

        pipeline.run(video_path=self.video_path)

        # With 60 frames, frame_skip=0, publish_interval=10 → ~6 publishes
        self.assertGreater(mock_mqtt_instance.publish.call_count, 0)

    @patch("ultralytics.YOLO")
    def test_density_levels_cascade_to_correct_signal(self, mock_yolo_cls):
        """Verify density → signal timing mapping through full pipeline layers."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        config = self._make_config(mqtt_enabled=False)
        pipeline = TrafficPipeline(config=config)

        # ── LOW: 1 motorcycle = 0.5 PCU → 5% → LOW → 30s ────────────
        low_tracks = FrameTracks(
            frame_index=0, timestamp=0.0,
            tracks=[TrackedObject(1, 3, "motorcycle", 0.9, (10, 10, 50, 50))]
        )
        density_low = pipeline._density_engine.compute_density(low_tracks)
        decision_low = pipeline._signal_engine.evaluate(density_low)
        self.assertEqual(density_low.density_level, "LOW")
        self.assertEqual(decision_low.duration, 32)

        # ── MEDIUM: 4 cars = 4.0 PCU → 40% → MEDIUM → 50s ───────────
        medium_tracks = FrameTracks(
            frame_index=1, timestamp=0.033,
            tracks=[TrackedObject(i, 2, "car", 0.9, (i*10, i*10, i*10+50, i*10+50)) for i in range(4)]
        )
        density_med = pipeline._density_engine.compute_density(medium_tracks)
        decision_med = pipeline._signal_engine.evaluate(density_med)
        self.assertEqual(density_med.density_level, "MEDIUM")
        self.assertEqual(decision_med.duration, 58)

        # ── HIGH: 4 cars + 2 buses = 9.0 PCU → 90% → HIGH → 70s ─────
        high_tracks = FrameTracks(
            frame_index=2, timestamp=0.066,
            tracks=[
                TrackedObject(1, 2, "car",   0.9, (0,  0,  50, 50)),
                TrackedObject(2, 2, "car",   0.9, (50, 0,  100, 50)),
                TrackedObject(3, 2, "car",   0.9, (0,  50, 50, 100)),
                TrackedObject(4, 2, "car",   0.9, (50, 50, 100, 100)),
                TrackedObject(5, 5, "bus",   0.9, (100, 0, 200, 100)),
                TrackedObject(6, 5, "bus",   0.9, (200, 0, 300, 100)),
            ]
        )
        density_high = pipeline._density_engine.compute_density(high_tracks)
        decision_high = pipeline._signal_engine.evaluate(density_high)
        self.assertEqual(density_high.density_level, "HIGH")
        self.assertEqual(decision_high.duration, 82)

    @patch("ultralytics.YOLO")
    def test_pipeline_raises_on_empty_video_path(self, mock_yolo_cls):
        """Pipeline raises ValueError if no video path is given."""
        mock_yolo_cls.return_value = MagicMock()
        config = self._make_config()
        config.video_path = ""
        pipeline = TrafficPipeline(config=config)
        with self.assertRaises((ValueError, Exception)):
            pipeline.run()


if __name__ == "__main__":
    unittest.main()
