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
    frame_skip: int = 2              # Process every Nth frame (0 = all frames)
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    # ── YOLO / Tracker ─────────────────────────────────────
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    tracker_type: str = "bytetrack.yaml"
    device: str = "cpu"

    # ── Density ────────────────────────────────────────────
    density: DensityConfig = field(default_factory=DensityConfig)

    # ── Signal ─────────────────────────────────────────────
    signal: SignalConfig = field(default_factory=SignalConfig)

    # ── MQTT ───────────────────────────────────────────────
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    mqtt_enabled: bool = True

    # ── Pipeline behaviour ─────────────────────────────────
    publish_interval_frames: int = 30   # Publish MQTT every N processed frames
    log_interval_frames: int = 30       # Log metrics every N processed frames
