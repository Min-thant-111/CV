"""
Data structures for traffic signal decision results.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SignalDecision:
    """Represents the output decision from the Signal Decision Engine."""

    signal: str  # Target signal state (e.g. "GREEN")
    duration: int  # Recommended green duration in seconds
    density_level: str  # "LOW", "MEDIUM", "HIGH"
    density_percentage: float
    vehicle_count: int
    reason: str
    road_path_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary representation."""
        return {
            "signal": self.signal,
            "duration": self.duration,
            "density_level": self.density_level,
            "density_percentage": round(self.density_percentage, 2),
            "vehicle_count": self.vehicle_count,
            "road_path_count": self.road_path_count,
            "reason": self.reason,
        }
