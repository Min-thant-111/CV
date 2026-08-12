"""Compare raw YOLO vehicle detections with ByteTrack output frame by frame.

This diagnostic is isolated from density estimation, MQTT, and traffic-signal
logic. It uses the existing YOLODetector and VehicleTracker implementations and
does not alter their production configuration.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.detection.yolo_detector import YOLODetector
from backend.models.detection import FrameDetections
from backend.models.tracking import FrameTracks
from backend.tracking.tracker import VehicleTracker
from backend.video.video_reader import VideoReader


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "debug_tracking.mp4"
CLASS_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "car": (60, 220, 60),
    "motorcycle": (255, 180, 30),
    "bus": (0, 165, 255),
    "truck": (60, 60, 255),
}


def parse_args() -> argparse.Namespace:
    """Parse options for one detector-vs-tracker comparison."""
    parser = argparse.ArgumentParser(
        description="Compare YOLO detections with ByteTrack tracks."
    )
    parser.add_argument("--video", required=True, help="Traffic video path.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum selected frames to process (default: 30, 0 = all).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based source frame at which comparison begins (default: 0).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth source frame (default: 1). Use 3 to match frame_skip=2.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights path or identifier (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Shared detector/tracker confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker configuration (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu, 0, or cuda (default: cpu).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Annotated output video (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or greater.")
    if args.start_frame < 0:
        parser.error("--start-frame must be 0 or greater.")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1.")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0.0 and 1.0.")

    return args


def selected_frames(
    reader: VideoReader,
    start_frame: int,
    frame_step: int,
    max_frames: int,
) -> Iterable[Tuple[int, float, object]]:
    """Yield the exact source-frame sequence supplied to both YOLO and ByteTrack."""
    selected = 0
    for frame_index, timestamp, frame in reader.read_frames():
        if frame_index < start_frame:
            continue
        if (frame_index - start_frame) % frame_step != 0:
            continue
        if max_frames and selected >= max_frames:
            break
        yield frame_index, timestamp, frame
        selected += 1


def class_counts(items: Iterable[object]) -> Counter:
    """Count objects exposing a ``class_name`` attribute."""
    return Counter(item.class_name for item in items)


def draw_tracks(
    frame,
    detections: FrameDetections,
    tracks: FrameTracks,
):
    """Draw tracked boxes/IDs and detector-vs-tracker totals."""
    annotated = frame.copy()

    for track in tracks.tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
        colour = CLASS_COLOURS.get(track.class_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        display_class = track.class_name.capitalize()
        label = f"{display_class} ID: {track.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_top = max(0, y1 - text_height - baseline - 5)
        label_left = max(
            0,
            min(x1, annotated.shape[1] - text_width - 7),
        )
        cv2.rectangle(
            annotated,
            (label_left, label_top),
            (min(annotated.shape[1] - 1, label_left + text_width + 6), y1),
            colour,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (label_left + 3, max(text_height, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    summary_lines = (
        f"Frame: {tracks.frame_index}",
        f"YOLO detections: {detections.count}",
        f"ByteTrack objects: {tracks.count}",
    )
    overlay_width = min(annotated.shape[1] - 1, 220)
    cv2.rectangle(
        annotated,
        (5, 5),
        (overlay_width, 72),
        (20, 20, 20),
        thickness=-1,
    )
    for line_index, line in enumerate(summary_lines):
        cv2.putText(
            annotated,
            line,
            (12, 25 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def print_frame_report(
    frame_index: int,
    detections: FrameDetections,
    tracks: FrameTracks,
) -> Dict[str, object]:
    """Print and return one requested frame comparison record."""
    detection_counts = class_counts(detections.detections)
    track_counts = class_counts(tracks.tracks)
    track_ids = sorted(set(tracks.get_track_ids()))

    record: Dict[str, object] = {
        "frame": frame_index,
        "yolo_detections": detections.count,
        "tracked_objects": tracks.count,
        "car_detections": detection_counts["car"],
        "car_tracks": track_counts["car"],
        "truck_detections": detection_counts["truck"],
        "truck_tracks": track_counts["truck"],
        "unique_track_ids": " ".join(str(track_id) for track_id in track_ids),
    }

    print(f"Frame: {frame_index}")
    print(f"YOLO detection count: {record['yolo_detections']}")
    print(f"Tracked object count: {record['tracked_objects']}")
    print(f"Car detections: {record['car_detections']}")
    print(f"Car tracks: {record['car_tracks']}")
    print(f"Truck detections: {record['truck_detections']}")
    print(f"Truck tracks: {record['truck_tracks']}")
    print(f"Unique track IDs: {track_ids}")
    print()

    return record


def write_csv(path: Path, records: List[Dict[str, object]]) -> None:
    """Save frame-level comparison records beside the debug video."""
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def run_debug_tracking(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Run independent YOLO detection and ByteTrack paths on identical frames."""
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
    csv_path = output_path.with_suffix(".csv")

    # Separate model instances prevent the raw predict calls from interacting
    # with the tracker's persistent internal state.
    raw_detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )
    tracking_detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )
    tracker = VehicleTracker(
        detector=tracking_detector,
        tracker_type=args.tracker,
    )

    output_fps = metadata.fps / args.frame_step
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open annotated output video: {output_path}")

    records: List[Dict[str, object]] = []
    seen_track_ids = set()
    try:
        for frame_index, timestamp, frame in selected_frames(
            reader,
            start_frame=args.start_frame,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
        ):
            detections = raw_detector.detect(frame, frame_index, timestamp)
            tracks = tracker.track_frame(frame, frame_index, timestamp)
            seen_track_ids.update(tracks.get_track_ids())

            records.append(print_frame_report(frame_index, detections, tracks))
            writer.write(draw_tracks(frame, detections, tracks))
    finally:
        writer.release()

    if not records:
        raise RuntimeError("No video frames were processed.")

    write_csv(csv_path, records)
    print(f"Selected frames processed: {len(records)}")
    print(f"Unique track IDs across sequence: {sorted(seen_track_ids)}")
    print(f"Annotated video: {output_path}")
    print(f"Frame comparison CSV: {csv_path}")
    return records


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        run_debug_tracking(args)
    except (ImportError, ModuleNotFoundError) as error:
        print(
            "ByteTrack dependency error. Ensure the 'lap' package is installed "
            f"for this Python environment: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
