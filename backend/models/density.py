"""
Data structures for traffic density estimation results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class DensityMetrics:
    """Represents calculated traffic density metrics for a video frame.

    Note: This metric represents a video-based computer vision traffic density
    estimation derived from active vehicle tracks within the camera field of view,
    rather than a physical inductive loop sensor measurement.
    """

    frame_index: int
    timestamp: float
    total_vehicle_count: int
    class_counts: Dict[str, int]
    weighted_vehicle_units: float
    capacity_units: float
    density_percentage: float
    density_level: str  # "LOW", "MEDIUM", "HIGH"
    active_track_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "total_vehicle_count": self.total_vehicle_count,
            "class_counts": self.class_counts,
            "weighted_vehicle_units": round(self.weighted_vehicle_units, 2),
            "capacity_units": round(self.capacity_units, 2),
            "density_percentage": round(self.density_percentage, 2),
            "density_level": self.density_level,
            "active_track_ids": self.active_track_ids,
        }
