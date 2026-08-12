"""Calibrate detected boxes into an estimated visible traffic demand."""

from dataclasses import dataclass
import math
from typing import Dict


@dataclass(frozen=True)
class VehicleCountEstimate:
    detected_count: int
    estimated_count: int
    class_counts: Dict[str, int]
    correction_factor: float


def estimate_visible_vehicles(
    detected_count: int,
    class_counts: Dict[str, int],
    frame_width: int,
    frame_height: int,
    road_path_count: int,
    traffic_level: str | None = None,
) -> VehicleCountEstimate:
    """Estimate vehicles missed through low resolution and dense occlusion.

    A 320x240 CCTV frame contains one quarter of the source pixels of the
    conservative 640x480 calibration resolution. Detection recall degrades
    approximately with linear resolution. Sparse scenes receive only 20% of
    that resolution correction (1.2x at 320x240), while a visibly crowded
    multi-path queue receives the full correction plus an occlusion allowance.

    A classifier-confirmed heavy queue is capped at 2.6x; geometry-only
    fallback remains capped at 2.4x. The raw detected count is always retained,
    so the UI can distinguish measurement from estimation.
    """
    detected = max(0, int(detected_count))
    paths = max(1, int(road_path_count))
    pixels = max(1, int(frame_width) * int(frame_height))
    reference_pixels = 640 * 480
    detected_per_path = detected / paths
    resolution_factor = min(
        2.0, max(1.0, math.sqrt(reference_pixels / pixels))
    )
    normalized_level = (traffic_level or "").strip().lower()
    crowded_queue = paths >= 3 and detected_per_path >= 5.0
    if normalized_level == "heavy":
        factor = min(2.6, resolution_factor * 1.3)
    elif not normalized_level and crowded_queue:
        factor = min(2.4, resolution_factor * 1.2)
    elif normalized_level == "medium":
        factor = 1.0 + (resolution_factor - 1.0) * 0.6
    else:
        # Free-flow vehicles are less occluded, but tiny distant vehicles can
        # still be missed. Apply only a small fraction of the resolution
        # correction so a sparse 320x240 scene is not doubled.
        factor = 1.0 + (resolution_factor - 1.0) * 0.2
    estimated = max(detected, int(round(detected * factor)))

    positive_counts = {
        name: max(0, int(count)) for name, count in class_counts.items()
    }
    observed_total = sum(positive_counts.values())
    if detected == 0 or observed_total == 0:
        estimated_classes = dict(positive_counts)
    else:
        exact = {
            name: estimated * count / observed_total
            for name, count in positive_counts.items()
        }
        estimated_classes = {name: int(value) for name, value in exact.items()}
        remainder = estimated - sum(estimated_classes.values())
        order = sorted(
            exact,
            key=lambda name: exact[name] - estimated_classes[name],
            reverse=True,
        )
        for name in order[:remainder]:
            estimated_classes[name] += 1

    return VehicleCountEstimate(
        detected_count=detected,
        estimated_count=estimated,
        class_counts=estimated_classes,
        correction_factor=round(factor, 3),
    )
