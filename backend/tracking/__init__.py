"""
Object tracking module (ByteTrack interface).
"""

from backend.tracking.tracker import VehicleTracker, TrackerError

__all__ = ["VehicleTracker", "TrackerError"]
