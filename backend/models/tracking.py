"""
Data structures for multi-object tracking results.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class TrackedObject:
    """Represents a single tracked vehicle with a persistent track ID."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict:
        """Convert tracked object to dictionary representation."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(c), 2) for c in self.bbox],
        }


@dataclass
class FrameTracks:
    """Represents all tracked vehicles within a single frame."""

    frame_index: int
    timestamp: float
    tracks: List[TrackedObject] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total number of tracked vehicles in the frame."""
        return len(self.tracks)

    def get_class_counts(self) -> Dict[str, int]:
        """Return breakdown count by vehicle class name."""
        counts: Dict[str, int] = {}
        for t in self.tracks:
            counts[t.class_name] = counts.get(t.class_name, 0) + 1
        return counts

    def get_track_ids(self) -> List[int]:
        """Return list of active track IDs in current frame."""
        return [t.track_id for t in self.tracks]
