"""Evaluate YOLO vehicle detections across fixed confidence thresholds.

The evaluator is deliberately isolated from the production pipeline. It does
not import or run tracking, density estimation, MQTT, or signal decisions.

YOLO inference runs once at the lowest threshold. Higher-threshold outputs are
derived by removing detections below each requested value. Because NMS handles
boxes in descending confidence order, lower-confidence boxes cannot suppress
the retained higher-confidence boxes; this produces the same retained set
without repeating identical model inference six times.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detection.debug_detection import draw_detections
from backend.detection.yolo_detector import DEFAULT_TARGET_CLASSES, YOLODetector
from backend.models.detection import FrameDetections
from backend.video.video_reader import VideoReader


THRESHOLDS: Tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
CLASS_NAMES: Tuple[str, ...] = tuple(DEFAULT_TARGET_CLASSES.values())
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "confidence_evaluation"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one confidence evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare YOLO vehicle counts at confidence 0.20 through 0.70."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the traffic video evaluated at every threshold.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to evaluate (default: 0, meaning all frames).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based source frame at which evaluation begins (default: 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights path or identifier (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu, 0, or cuda (default: cpu).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for annotated videos and CSV (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or greater.")
    if args.start_frame < 0:
        parser.error("--start-frame must be 0 or greater.")

    return args


def threshold_slug(threshold: float) -> str:
    """Return a filesystem-safe threshold label, such as ``0_20``."""
    return f"{threshold:.2f}".replace(".", "_")


def filtered_detections(
    detections: FrameDetections, threshold: float
) -> FrameDetections:
    """Retain detections whose confidence meets the selected threshold."""
    return FrameDetections(
        frame_index=detections.frame_index,
        timestamp=detections.timestamp,
        detections=[
            detection
            for detection in detections.detections
            if detection.confidence >= threshold
        ],
    )


def create_video_writer(
    path: Path, fps: float, size: Tuple[int, int]
) -> cv2.VideoWriter:
    """Create and validate one MP4 output writer."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open annotated output video: {path}")
    return writer


def write_summary_csv(
    path: Path, totals: Dict[float, Counter]
) -> None:
    """Save aggregate per-threshold counts for later comparison."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["threshold", "total", *CLASS_NAMES],
        )
        writer.writeheader()
        for threshold in THRESHOLDS:
            counts = totals[threshold]
            writer.writerow(
                {
                    "threshold": f"{threshold:.2f}",
                    "total": counts["total"],
                    **{name: counts[name] for name in CLASS_NAMES},
                }
            )


def print_summary(totals: Dict[float, Counter], processed: int) -> None:
    """Print aggregate class counts in a compact table."""
    print(f"Frames evaluated: {processed}")
    print()
    print(
        f"{'Threshold':>9} | {'Total':>7} | {'Car':>7} | "
        f"{'Motorcycle':>10} | {'Bus':>7} | {'Truck':>7}"
    )
    print("-" * 67)
    for threshold in THRESHOLDS:
        counts = totals[threshold]
        print(
            f"{threshold:>9.2f} | {counts['total']:>7} | "
            f"{counts['car']:>7} | {counts['motorcycle']:>10} | "
            f"{counts['bus']:>7} | {counts['truck']:>7}"
        )


def run_evaluation(args: argparse.Namespace) -> Dict[float, Counter]:
    """Run YOLO-only evaluation and produce all threshold artifacts."""
    reader = VideoReader(args.video)
    metadata = reader.metadata
    if metadata is None:
        raise RuntimeError("Video metadata was not available after opening the video.")
    if args.start_frame >= metadata.total_frames:
        raise ValueError(
            f"--start-frame {args.start_frame} is outside the video "
            f"({metadata.total_frames} frames)."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=min(THRESHOLDS),
        device=args.device,
    )
    totals: Dict[float, Counter] = {
        threshold: Counter({name: 0 for name in (*CLASS_NAMES, "total")})
        for threshold in THRESHOLDS
    }

    video_paths = {
        threshold: output_dir / f"confidence_{threshold_slug(threshold)}.mp4"
        for threshold in THRESHOLDS
    }
    writers: Dict[float, cv2.VideoWriter] = {}
    processed = 0

    try:
        with ExitStack() as stack:
            for threshold, path in video_paths.items():
                writer = create_video_writer(
                    path,
                    metadata.fps,
                    (metadata.width, metadata.height),
                )
                writers[threshold] = writer
                stack.callback(writer.release)

            for frame_index, timestamp, frame in reader.read_frames():
                if frame_index < args.start_frame:
                    continue
                if args.max_frames and processed >= args.max_frames:
                    break

                base_detections = detector.detect(
                    frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )

                for threshold in THRESHOLDS:
                    selected = filtered_detections(base_detections, threshold)
                    frame_counts = Counter(
                        detection.class_name for detection in selected.detections
                    )
                    for class_name in CLASS_NAMES:
                        totals[threshold][class_name] += frame_counts[class_name]
                    totals[threshold]["total"] += selected.count

                    annotated = draw_detections(frame, selected)
                    cv2.putText(
                        annotated,
                        f"Confidence >= {threshold:.2f}",
                        (8, max(58, annotated.shape[0] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writers[threshold].write(annotated)

                processed += 1
    finally:
        # ExitStack normally releases all writers. This also covers failures
        # that occur while creating writers before they enter the stack.
        for writer in writers.values():
            writer.release()

    if processed == 0:
        raise RuntimeError("No video frames were processed.")

    summary_path = output_dir / "confidence_summary.csv"
    write_summary_csv(summary_path, totals)
    print_summary(totals, processed)
    print()
    print(f"Summary CSV: {summary_path}")
    for threshold in THRESHOLDS:
        print(f"Confidence {threshold:.2f} video: {video_paths[threshold]}")

    return totals


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        run_evaluation(args)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
