"""
Traffic Density Estimation Engine.

Calculates video-based traffic density metrics from tracked vehicle objects
using a Passenger Car Unit (PCU) weighted capacity model and configurable
classification thresholds (LOW, MEDIUM, HIGH).

Note:
This module provides a computer vision-based traffic density estimation derived
from tracked vehicle visual features in the camera field of view (or ROI zone),
rather than a physical inductive loop sensor measurement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.models.tracking import FrameTracks, TrackedObject
from backend.models.density import DensityMetrics


DEFAULT_VEHICLE_WEIGHTS: Dict[str, float] = {
    "motorcycle": 0.5,
    "car": 1.0,
    "bus": 2.5,
    "truck": 3.0,
}


@dataclass
class DensityConfig:
    """Configurable parameters for traffic density estimation."""

    vehicle_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VEHICLE_WEIGHTS)
    )
    max_road_capacity_units: float = 10.0  # PCU capacity per road path
    low_threshold_pct: float = 45.0        # <= 45% is LOW density
    high_threshold_pct: float = 70.0       # >= 70% is HIGH density (>45%-<70% is MEDIUM)
    five_path_low_vehicle_limit: int = 22  # Count guardrail for wide five-path roads
    roi_polygon: Optional[List[Tuple[float, float]]] = None  # Polygon coordinates for ROI filtering
    road_path_count: int = 1               # Parallel paths/ways visible in the road


class DensityEngineError(Exception):
    """Base exception for DensityEngine errors."""

    pass


class DensityEngine:
    """Deterministic Traffic Density Estimation Engine."""

    def __init__(self, config: Optional[DensityConfig] = None):
        """Args:

        config: DensityConfig instance holding custom weights and thresholds.
        """
        self.config = config or DensityConfig()

    def set_roi_polygon(self, polygon: Optional[List[Tuple[float, float]]]) -> None:
        """Configure optional Region of Interest (ROI) polygon."""
        self.config.roi_polygon = polygon

    def _is_inside_roi(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if vehicle bounding box center falls within configured ROI polygon."""
        if not self.config.roi_polygon or len(self.config.roi_polygon) < 3:
            return True  # If no ROI defined, count all objects in frame

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        pts = np.array(self.config.roi_polygon, dtype=np.int32)
        res = cv2.pointPolygonTest(pts, (float(center_x), float(center_y)), False)
        return res >= 0

    def compute_density(self, frame_tracks: FrameTracks) -> DensityMetrics:
        """Compute traffic density metrics for a single frame track set.

        Args:
            frame_tracks: FrameTracks object containing active vehicle tracks.

        Returns:
            DensityMetrics object containing counts, weighted PCU units, density percentage, and level.
        """
        if frame_tracks is None:
            raise ValueError("Invalid FrameTracks provided to DensityEngine.")

        filtered_tracks: List[TrackedObject] = []
        for track in frame_tracks.tracks:
            if self._is_inside_roi(track.bbox):
                filtered_tracks.append(track)

        # Count breakdown by vehicle class
        class_counts: Dict[str, int] = {}
        active_track_ids: List[int] = []

        for track in filtered_tracks:
            cls = track.class_name
            class_counts[cls] = class_counts.get(cls, 0) + 1
            active_track_ids.append(track.track_id)

        return self.compute_density_from_counts(
            class_counts,
            frame_index=frame_tracks.frame_index,
            timestamp=frame_tracks.timestamp,
            active_track_ids=active_track_ids,
        )

    def compute_density_from_counts(
        self,
        class_counts: Dict[str, int],
        frame_index: int = 0,
        timestamp: float = 0.0,
        active_track_ids: Optional[List[int]] = None,
    ) -> DensityMetrics:
        """Compute density from measured or calibrated per-class counts."""
        normalized_counts = {
            name: max(0, int(count))
            for name, count in (class_counts or {}).items()
            if int(count) > 0
        }
        total_count = sum(normalized_counts.values())
        weighted_units = sum(
            count * self.config.vehicle_weights.get(name, 1.0)
            for name, count in normalized_counts.items()
        )

        # 3. Each parallel road path adds usable capacity.  Do not clamp the
        # demand ratio to 100%; otherwise, for example, 30 cars on one path and
        # 30 cars on three paths both collapse to 100% and become indistinguishable.
        road_path_count = max(1, int(self.config.road_path_count))
        capacity_per_path = max(0.1, self.config.max_road_capacity_units)
        capacity = capacity_per_path * road_path_count
        density_pct = (weighted_units / capacity) * 100.0
        density_pct = round(density_pct, 2)

        # 4. Classify density level (LOW, MEDIUM, HIGH)
        five_path_count_is_low = (
            road_path_count >= 5
            and total_count <= max(0, int(self.config.five_path_low_vehicle_limit))
        )
        if density_pct <= self.config.low_threshold_pct or five_path_count_is_low:
            density_level = "LOW"
        elif density_pct < self.config.high_threshold_pct:
            density_level = "MEDIUM"
        else:
            density_level = "HIGH"

        return DensityMetrics(
            frame_index=frame_index,
            timestamp=timestamp,
            total_vehicle_count=total_count,
            class_counts=normalized_counts,
            weighted_vehicle_units=round(weighted_units, 2),
            capacity_units=capacity,
            density_percentage=density_pct,
            density_level=density_level,
            active_track_ids=active_track_ids or [],
            road_path_count=road_path_count,
        )
