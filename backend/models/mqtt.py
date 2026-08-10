"""
Data structures for MQTT messages and payloads.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MQTTSignalPayload:
    """Represents the telemetry and signal decision payload sent over MQTT."""

    intersection_id: int
    signal: str
    duration: int
    density: float
    density_level: str
    vehicle_count: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to JSON-serializable dictionary format."""
        return {
            "intersection": self.intersection_id,
            "signal": self.signal,
            "duration": self.duration,
            "density": round(self.density, 2),
            "density_level": self.density_level,
            "vehicle_count": self.vehicle_count,
            "timestamp": round(self.timestamp, 3),
        }
