"""Standalone YOLO vehicle-detection diagnostics for traffic videos.

This module intentionally runs detection only. It does not use tracking,
density estimation, MQTT, or traffic-signal decision logic.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import cv2


# Support both ``python -m backend.detection.debug_detection`` and direct use.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detection.yolo_detector import DEFAULT_TARGET_CLASSES, YOLODetector
from backend.models.detection import FrameDetections
from backend.video.video_reader import VideoReader


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "debug_detection.mp4"

# OpenCV BGR colours provide a distinct visual identity for each vehicle class.
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "car": (60, 220, 60),
    "motorcycle": (255, 180, 30),
    "bus": (0, 165, 255),
    "truck": (60, 60, 255),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a single diagnostic run."""
    parser = argparse.ArgumentParser(
        description="Run YOLO-only vehicle detection and save an annotated video."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the traffic video to inspect.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to process (default: 300, 0 = all).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based source frame at which diagnostics begin (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights path or model identifier (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu, 0, or cuda (default: cpu).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Annotated output path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show annotated frames in a live OpenCV window; press Q to stop.",
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or greater.")
    if args.start_frame < 0:
        parser.error("--start-frame must be 0 or greater.")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0.0 and 1.0.")

    return args


def draw_detections(frame, frame_detections: FrameDetections):
    """Draw vehicle boxes, class/confidence/coordinates, and frame total."""
    annotated = frame.copy()

    for detection in frame_detections.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        colour = CLASS_COLOURS.get(detection.class_name, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        label = (
            f"{detection.class_name} {detection.confidence:.2f} "
            f"[{x1},{y1},{x2},{y2}]"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        )
        label_top = max(0, y1 - text_height - baseline - 5)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(annotated.shape[1] - 1, x1 + text_width + 6), y1),
            colour,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(text_height, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    total_label = f"Total detections: {frame_detections.count}"
    cv2.rectangle(annotated, (8, 8), (235, 38), (20, 20, 20), thickness=-1)
    cv2.putText(
        annotated,
        total_label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def print_frame_summary(frame_detections: FrameDetections) -> None:
    """Print a stable per-class detection summary for one source frame."""
    counts = Counter(
        detection.class_name for detection in frame_detections.detections
    )

    print(f"Frame: {frame_detections.frame_index}")
    for class_name in DEFAULT_TARGET_CLASSES.values():
        print(f"{class_name}: {counts.get(class_name, 0)}")
    print(f"total: {frame_detections.count}")
    print()


def run_debug_detection(args: argparse.Namespace) -> int:
    """Execute detector-only inference and write the annotated diagnostic video."""
    reader = VideoReader(args.video)
    metadata = reader.metadata
    if metadata is None:
        raise RuntimeError("Video metadata was not available after opening the video.")

    if args.start_frame >= metadata.total_frames:
        raise ValueError(
            f"--start-frame {args.start_frame} is outside the video "
            f"({metadata.total_frames} frames)."
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open annotated output video: {output_path}")

    processed = 0
    try:
        for frame_index, timestamp, frame in reader.read_frames():
            if frame_index < args.start_frame:
                continue
            if args.max_frames and processed >= args.max_frames:
                break

            frame_detections = detector.detect(
                frame,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            annotated = draw_detections(frame, frame_detections)
            writer.write(annotated)
            print_frame_summary(frame_detections)
            processed += 1

            if args.show:
                cv2.imshow("YOLO Detection Debug", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed frames: {processed}")
    print(f"Annotated video: {output_path}")
    return processed


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        run_debug_detection(args)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
