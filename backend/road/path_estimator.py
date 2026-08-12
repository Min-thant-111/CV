"""Estimate the number of visible drivable road paths from video imagery.

The estimator samples several frames and looks for persistent lane/road
boundaries that converge with perspective.  It is intentionally independent
of vehicle detection so it adds very little startup cost to video analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RoadPathEstimate:
    path_count: int
    confidence: float
    sampled_frames: int
    method: str = "multi-frame lane-boundary detection"


class RoadPathEstimator:
    """Infer 1-12 road paths from persistent perspective-aligned boundaries."""

    def __init__(self, max_paths: int = 5, sample_count: int = 12):
        self.max_paths = max(1, int(max_paths))
        self.sample_count = max(3, int(sample_count))

    @staticmethod
    def _cluster_positions(positions: List[float], tolerance: float) -> List[float]:
        if not positions:
            return []
        clusters: List[List[float]] = [[value] for value in sorted(positions)]
        merged = True
        while merged:
            merged = False
            compacted: List[List[float]] = []
            for cluster in clusters:
                if (
                    compacted
                    and abs(np.mean(cluster) - np.mean(compacted[-1])) <= tolerance
                ):
                    compacted[-1].extend(cluster)
                    merged = True
                else:
                    compacted.append(cluster)
            clusters = compacted
        return [float(np.median(cluster)) for cluster in clusters]

    def estimate_frame(self, frame: np.ndarray) -> Tuple[int, float]:
        """Return ``(path_count, confidence)`` for one BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 1, 0.0

        height, width = frame.shape[:2]
        if height < 40 or width < 40:
            return 1, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 55, 155)

        # Road boundaries are most separable in the lower portion of a fixed
        # traffic camera. Ignore sky, signs, and most building edges.
        roi_top = int(height * 0.28)
        edges[:roi_top, :] = 0
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(24, width // 35),
            minLineLength=max(24, height // 8),
            maxLineGap=max(14, height // 18),
        )
        if lines is None:
            return 1, 0.0

        reference_y = height * 0.88
        positions: List[float] = []
        for raw_line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in raw_line)
            dx, dy = x2 - x1, y2 - y1
            if abs(dy) < height * 0.10:
                continue
            # Reject near-horizontal texture while retaining vertical and
            # perspective-diagonal lane/road boundaries.
            if abs(dy) < abs(dx) * 0.55:
                continue
            x_at_reference = x1 + (reference_y - y1) * dx / dy
            if -0.08 * width <= x_at_reference <= 1.08 * width:
                positions.append(x_at_reference)

        boundaries = self._cluster_positions(positions, tolerance=width * 0.045)
        if len(boundaries) < 2:
            return 1, 0.0

        # Visible roadway edges plus internal separators define N-1 spaces.
        path_count = min(self.max_paths, max(1, len(boundaries) - 1))
        line_support = min(1.0, len(positions) / max(4.0, len(boundaries) * 1.5))
        span = (max(boundaries) - min(boundaries)) / width
        confidence = min(1.0, 0.55 * line_support + 0.45 * min(1.0, span / 0.55))
        return path_count, round(confidence, 3)

    def estimate_video(self, video_path: str) -> RoadPathEstimate:
        """Sample frames throughout a video and return the consensus estimate."""
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return RoadPathEstimate(1, 0.0, 0, "fallback: unreadable video")

        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_indices = np.linspace(
            0, max(0, total_frames - 1), min(self.sample_count, total_frames), dtype=int
        )
        observations: List[Tuple[int, float]] = []
        sampled = 0
        for frame_index in sorted(set(int(index) for index in frame_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            read_ok, frame = capture.read()
            if not read_ok:
                continue
            sampled += 1
            count, confidence = self.estimate_frame(frame)
            if confidence >= 0.35:
                observations.append((count, confidence))
        capture.release()

        if not observations:
            return RoadPathEstimate(1, 0.0, sampled, "fallback: no stable boundaries")

        scores = {}
        for count, confidence in observations:
            scores[count] = scores.get(count, 0.0) + confidence
        path_count = max(scores, key=lambda count: (scores[count], count))
        agreement = sum(
            confidence for count, confidence in observations if count == path_count
        ) / max(0.001, sum(confidence for _, confidence in observations))
        coverage = len(observations) / max(1, sampled)
        confidence = round(min(1.0, 0.65 * agreement + 0.35 * coverage), 3)
        return RoadPathEstimate(path_count, confidence, sampled)
