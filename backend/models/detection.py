"""
Data structures for object detection results.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class Detection:
    """Represents a single detected vehicle in a frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict:
        """Convert detection to dictionary representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
        }


@dataclass
class FrameDetections:
    """Represents all vehicle detections within a single frame."""

    frame_index: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total number of detected vehicles in frame."""
        return len(self.detections)

    def get_class_counts(self) -> Dict[str, int]:
        """Return breakdown count by vehicle class name."""
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts
