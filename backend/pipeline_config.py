"""
Pipeline configuration dataclass combining all module configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
from backend.density.density_engine import DensityConfig
from backend.signaling.signal_engine import SignalConfig
from backend.mqtt.mqtt_publisher import MQTTConfig


@dataclass
class PipelineConfig:
    """Top-level configuration for the full Edge-CV traffic analysis pipeline."""

    # ── Video ──────────────────────────────────────────────
    video_path: str = ""
    frame_skip: int = 1              # Process at 5 FPS for typical 10 FPS CCTV
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    # ── YOLO / Tracker ─────────────────────────────────────
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.20
    inference_size: int = 640
    iou_threshold: float = 0.45
    high_recall_tiling: bool = True
    tile_inference_size: int = 640
    tile_confidence_threshold: float = 0.18
    tile_grid_size: int = 3
    tile_interval_frames: int = 5
    detection_memory_frames: int = 10
    class_history_frames: int = 12
    heavy_vehicle_min_confidence: float = 0.30
    heavy_vehicle_min_observations: int = 3
    class_switch_margin: float = 1.20
    suppress_camera_overlay: bool = True
    overlay_top_fraction: float = 0.24
    overlay_left_fraction: float = 0.50
    tracker_type: str = "bytetrack.yaml"
    device: Optional[str] = None

    # ── Density ────────────────────────────────────────────
    density: DensityConfig = field(default_factory=DensityConfig)

    # ── Signal ─────────────────────────────────────────────
    signal: SignalConfig = field(default_factory=SignalConfig)

    # ── MQTT ───────────────────────────────────────────────
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    mqtt_enabled: bool = True

    # ── Pipeline behaviour ─────────────────────────────────
    publish_interval_frames: int = 5    # Publish MQTT every N processed frames
    log_interval_frames: int = 5        # Log metrics every N processed frames
