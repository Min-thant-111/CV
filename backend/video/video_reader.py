"""
Video Reader module for OpenCV-based video ingestion and metadata extraction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple, Optional
import cv2
import numpy as np


# Allowed/Supported video extensions
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class VideoMetadata:
    """Dataclass holding video properties and metadata."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float


class VideoReaderError(Exception):
    """Base exception class for VideoReader errors."""

    pass


class VideoNotFoundError(VideoReaderError, FileNotFoundError):
    """Raised when the specified video file does not exist."""

    pass


class InvalidVideoError(VideoReaderError, ValueError):
    """Raised when the video file is corrupt, unreadable, or invalid."""

    pass


class VideoReader:
    """OpenCV Video Capture wrapper providing safe validation, metadata extraction,

    and low-memory frame iteration.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path).resolve()
        self.metadata: Optional[VideoMetadata] = None

        self._validate_and_extract_metadata()

    def _validate_and_extract_metadata(self) -> None:
        """Validate that the video file exists and can be opened by OpenCV,

        extracting video metadata attributes.
        """
        if not self.video_path.exists():
            raise VideoNotFoundError(
                f"Video file not found at path: '{self.video_path}'"
            )

        if not self.video_path.is_file():
            raise InvalidVideoError(
                f"Specified path is not a file: '{self.video_path}'"
            )

        ext = self.video_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidVideoError(
                f"Unsupported video file extension '{ext}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            raise InvalidVideoError(
                f"Failed to open video file with OpenCV: '{self.video_path}'"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or total_frames <= 0:
            cap.release()
            raise InvalidVideoError(
                f"Invalid or corrupt video metadata: width={width}, height={height}, total_frames={total_frames}"
            )

        # Fallback for FPS if unreadable or 0
        if fps <= 0:
            fps = 30.0

        duration_seconds = total_frames / fps

        self.metadata = VideoMetadata(
            path=str(self.video_path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=round(duration_seconds, 2),
        )

        cap.release()

    def read_frames(
        self,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield frames sequentially without loading the full video into memory.

        Yields:
            Tuple[frame_index, timestamp_seconds, frame_bgr_ndarray]
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise InvalidVideoError(
                f"Unable to open video capture stream for: {self.video_path}"
            )

        frame_index = 0
        fps = self.metadata.fps if self.metadata else 30.0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp = round(frame_index / fps, 3)
                yield frame_index, timestamp, frame
                frame_index += 1
        finally:
            cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
