"""
Video Processor module for frame preprocessing and streaming interface.
"""

from typing import Generator, Tuple, Optional
import cv2
import numpy as np
from backend.video.video_reader import VideoReader, VideoMetadata


class VideoProcessor:
    """Processes video frames from VideoReader, supporting frame resizing,

    frame skipping for real-time edge CV, and structured frame delivery.
    """

    def __init__(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        frame_skip: int = 0,
    ):
        """Args:

        video_path: Path to local video file.
        target_size: Optional (width, height) tuple to resize frames for YOLO input.
        frame_skip: Number of frames to skip between yields (0 = process every frame).
        """
        self.reader = VideoReader(video_path)
        self.target_size = target_size
        self.frame_skip = max(0, frame_skip)

    @property
    def metadata(self) -> VideoMetadata:
        """Return metadata of the underlying video."""
        return self.reader.metadata

    def process_frames(
        self,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator yielding processed frames ready for object detection.

        Yields:
            Tuple[frame_index, timestamp_seconds, processed_frame_ndarray]
        """
        for frame_index, timestamp, frame in self.reader.read_frames():
            # Handle frame skipping if configured
            if self.frame_skip > 0 and (frame_index % (self.frame_skip + 1) != 0):
                continue

            # Optional frame resize for YOLO or downsampling
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

            yield frame_index, timestamp, frame
