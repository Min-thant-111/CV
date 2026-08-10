"""
Unit tests for VideoReader and VideoProcessor modules.
"""

import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from backend.video.video_reader import (
    VideoReader,
    VideoNotFoundError,
    InvalidVideoError,
    VideoMetadata,
)
from backend.video.video_processor import VideoProcessor


def create_synthetic_test_video(
    output_path: str, width: int = 320, height: int = 240, fps: int = 30, frame_count: int = 15
) -> str:
    """Helper function to generate a temporary valid video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frame_count):
        # Create dummy frame with changing color
        frame = np.full((height, width, 3), (i * 15 % 255, 100, 150), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return output_path


class TestVideoModule(unittest.TestCase):
    """Test suite for video validation, metadata extraction, and frame streaming."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_video_path = os.path.join(self.temp_dir.name, "sample_test.mp4")
        create_synthetic_test_video(
            self.valid_video_path, width=320, height=240, fps=30, frame_count=15
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nonexistent_video_throws_exception(self):
        """Test that attempting to read a non-existent file raises VideoNotFoundError."""
        fake_path = os.path.join(self.temp_dir.name, "nonexistent.mp4")
        with self.assertRaises(VideoNotFoundError):
            VideoReader(fake_path)

    def test_unsupported_extension_throws_exception(self):
        """Test that an invalid file extension raises InvalidVideoError."""
        invalid_ext_path = os.path.join(self.temp_dir.name, "file.txt")
        Path(invalid_ext_path).write_text("not a video")
        with self.assertRaises(InvalidVideoError):
            VideoReader(invalid_ext_path)

    def test_metadata_extraction(self):
        """Test accurate extraction of width, height, fps, total_frames, and duration."""
        reader = VideoReader(self.valid_video_path)
        meta = reader.metadata

        self.assertIsInstance(meta, VideoMetadata)
        self.assertEqual(meta.width, 320)
        self.assertEqual(meta.height, 240)
        self.assertEqual(meta.fps, 30.0)
        self.assertEqual(meta.total_frames, 15)
        self.assertAlmostEqual(meta.duration_seconds, 0.5, places=2)

    def test_frame_generator_iteration(self):
        """Test that read_frames yields frames frame-by-frame as numpy arrays."""
        reader = VideoReader(self.valid_video_path)
        frames_list = []

        for frame_idx, timestamp, frame in reader.read_frames():
            self.assertIsInstance(frame_idx, int)
            self.assertIsInstance(timestamp, float)
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (240, 320, 3))
            frames_list.append(frame_idx)

        self.assertEqual(len(frames_list), 15)
        self.assertEqual(frames_list, list(range(15)))

    def test_video_processor_resizing_and_skipping(self):
        """Test VideoProcessor frame target sizing and frame skipping."""
        processor = VideoProcessor(
            self.valid_video_path, target_size=(160, 120), frame_skip=1
        )

        processed_frames = list(processor.process_frames())
        # With 15 frames total and frame_skip=1 (skipping every 2nd frame), we expect 8 frames (0, 2, 4, 6, 8, 10, 12, 14)
        self.assertEqual(len(processed_frames), 8)

        # Check resized frame dimensions
        _, _, frame = processed_frames[0]
        self.assertEqual(frame.shape, (120, 160, 3))


if __name__ == "__main__":
    unittest.main()
