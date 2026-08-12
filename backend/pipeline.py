"""
Edge-CV Traffic Analysis Pipeline.

Orchestrates all modules in strict layer separation:
  Video → YOLO → ByteTrack → Density → Signal → MQTT
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from backend.pipeline_config import PipelineConfig
from backend.video.video_processor import VideoProcessor
from backend.detection.yolo_detector import YOLODetector
from backend.tracking.tracker import VehicleTracker
from backend.density.density_engine import DensityEngine
from backend.signaling.signal_engine import SignalDecisionEngine
from backend.mqtt.mqtt_publisher import MQTTPublisher
from backend.models.density import DensityMetrics
from backend.models.signaling import SignalDecision
from backend.models.tracking import FrameTracks
from backend.road.path_estimator import RoadPathEstimator

logger = logging.getLogger("TrafficPipeline")


@dataclass
class FrameResult:
    """Structured result for a single processed pipeline frame interval."""

    frame_index: int
    timestamp: float
    vehicle_count: int
    class_counts: Dict[str, int]
    density_percentage: float
    density_level: str
    signal: str
    green_duration: int
    mqtt_published: bool
    reason: str = ""

    def log_summary(self) -> None:
        """Write a single structured log line for this frame result."""
        logger.info(
            f"Frame {self.frame_index:>6} | "
            f"t={self.timestamp:>7.2f}s | "
            f"Vehicles={self.vehicle_count:>3} "
            f"[car={self.class_counts.get('car', 0)} "
            f"moto={self.class_counts.get('motorcycle', 0)} "
            f"bus={self.class_counts.get('bus', 0)} "
            f"truck={self.class_counts.get('truck', 0)}] | "
            f"Density={self.density_percentage:>5.1f}% ({self.density_level:<6}) | "
            f"Signal={self.signal} {self.green_duration}s | "
            f"MQTT={'OK' if self.mqtt_published else 'SKIP'}"
        )


class TrafficPipeline:
    """Complete modular Edge-CV traffic analysis pipeline.

    Architecture (strict layer separation):
      Video Module     → raw BGR frames
      Detector Module  → per-frame vehicle detections
      Tracker Module   → persistent cross-frame track IDs
      Density Module   → traffic density metrics
      Signal Module    → green duration decision
      MQTT Module      → IoT dispatch (boundary to controller)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # ── Detector ────────────────────────────────────────────────
        target_size: Optional[Tuple[int, int]] = None
        if self.config.target_width and self.config.target_height:
            target_size = (self.config.target_width, self.config.target_height)

        self._detector = YOLODetector(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
            inference_size=self.config.inference_size,
            iou_threshold=self.config.iou_threshold,
        )

        # ── Tracker ─────────────────────────────────────────────────
        self._tracker = VehicleTracker(
            detector=self._detector,
            tracker_type=self.config.tracker_type,
            high_recall=self.config.high_recall_tiling,
            tile_inference_size=self.config.tile_inference_size,
            tile_confidence_threshold=self.config.tile_confidence_threshold,
            tile_grid_size=self.config.tile_grid_size,
            tile_interval_frames=self.config.tile_interval_frames,
            far_field_recall=self.config.far_field_recall,
            far_field_inference_size=self.config.far_field_inference_size,
            far_field_confidence_threshold=self.config.far_field_confidence_threshold,
            detection_memory_frames=self.config.detection_memory_frames,
            class_history_frames=self.config.class_history_frames,
            heavy_vehicle_min_confidence=self.config.heavy_vehicle_min_confidence,
            heavy_vehicle_min_observations=self.config.heavy_vehicle_min_observations,
            class_switch_margin=self.config.class_switch_margin,
            suppress_camera_overlay=self.config.suppress_camera_overlay,
            overlay_top_fraction=self.config.overlay_top_fraction,
            overlay_left_fraction=self.config.overlay_left_fraction,
        )

        # ── Density Engine ──────────────────────────────────────────
        self._density_engine = DensityEngine(config=self.config.density)

        # ── Signal Decision Engine ──────────────────────────────────
        self._signal_engine = SignalDecisionEngine(config=self.config.signal)

        # ── MQTT Publisher ──────────────────────────────────────────
        self._mqtt_publisher = MQTTPublisher(config=self.config.mqtt)
        self._mqtt_connected = False

        logger.info("TrafficPipeline initialised successfully.")

    # ─────────────────────────── MQTT Lifecycle ─────────────────────

    def connect_mqtt(self) -> bool:
        """Attempt MQTT broker connection. Non-fatal if broker is unavailable."""
        if not self.config.mqtt_enabled:
            logger.info("MQTT disabled by configuration.")
            return False
        self._mqtt_connected = self._mqtt_publisher.connect()
        return self._mqtt_connected

    def disconnect_mqtt(self) -> None:
        """Disconnect from MQTT broker cleanly."""
        self._mqtt_publisher.disconnect()
        self._mqtt_connected = False

    # ─────────────────────────── Frame Processing ───────────────────

    def _process_frame(
        self,
        frame,
        frame_index: int,
        timestamp: float,
    ) -> Tuple[FrameTracks, DensityMetrics, SignalDecision]:
        """Run one frame through tracker → density → signal pipeline layers.

        Video processing does not touch MQTT.
        YOLO does not control traffic signals.
        Density engine does not publish MQTT messages.
        Signal engine does not directly call the controller.
        """

        # Layer 1 – Track vehicles (YOLO + ByteTrack)
        frame_tracks: FrameTracks = self._tracker.track_frame(
            frame, frame_index=frame_index, timestamp=timestamp
        )

        # Layer 2 – Estimate traffic density
        density: DensityMetrics = self._density_engine.compute_density(frame_tracks)

        # Layer 3 – Determine signal timing
        decision: SignalDecision = self._signal_engine.evaluate(density)

        return frame_tracks, density, decision

    def _maybe_publish(
        self,
        decision: SignalDecision,
        frame_index: int,
    ) -> bool:
        """Publish MQTT signal decision at configured intervals.

        MQTT is the only communication boundary between the AI pipeline and the controller.
        """
        if not self.config.mqtt_enabled:
            return False
        if frame_index % self.config.publish_interval_frames != 0:
            return False
        return self._mqtt_publisher.publish_signal(decision)

    # ─────────────────────────── Public API ─────────────────────────

    def run(self, video_path: Optional[str] = None) -> None:
        """Execute the full pipeline on the specified video file.

        Args:
            video_path: Path to the input traffic video. Overrides config if provided.
        """
        path = video_path or self.config.video_path
        if not path:
            raise ValueError("No video path specified. Provide video_path in config or as argument.")

        target_size: Optional[Tuple[int, int]] = None
        if self.config.target_width and self.config.target_height:
            target_size = (self.config.target_width, self.config.target_height)

        processor = VideoProcessor(
            video_path=path,
            target_size=target_size,
            frame_skip=self.config.frame_skip,
        )

        meta = processor.metadata
        path_estimate = RoadPathEstimator().estimate_video(path)
        self.config.density.road_path_count = path_estimate.path_count
        self._density_engine.config.road_path_count = path_estimate.path_count
        logger.info(
            f"Video loaded: {meta.path} | "
            f"{meta.width}x{meta.height} | "
            f"{meta.fps:.1f} FPS | "
            f"{meta.total_frames} frames | "
            f"{meta.duration_seconds:.1f}s"
        )
        logger.info(
            "Automatically detected %d road path(s) (confidence %.1f%%).",
            path_estimate.path_count,
            path_estimate.confidence * 100,
        )

        self.connect_mqtt()

        processed = 0

        try:
            for frame_index, timestamp, frame in processor.process_frames():
                # ── Core pipeline layers (strictly separated) ──────
                frame_tracks, density, decision = self._process_frame(
                    frame, frame_index, timestamp
                )

                processed += 1

                # ── MQTT publish at controlled interval ────────────
                mqtt_ok = self._maybe_publish(decision, processed)

                # ── Structured logging at configured interval ──────
                if processed % self.config.log_interval_frames == 0:
                    result = FrameResult(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        vehicle_count=density.total_vehicle_count,
                        class_counts=density.class_counts,
                        density_percentage=density.density_percentage,
                        density_level=density.density_level,
                        signal=decision.signal,
                        green_duration=decision.duration,
                        mqtt_published=mqtt_ok,
                        reason=decision.reason,
                    )
                    result.log_summary()

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        finally:
            self.disconnect_mqtt()
            logger.info(f"Pipeline complete. Total frames processed: {processed}")
