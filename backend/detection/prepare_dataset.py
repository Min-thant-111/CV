"""Prepare manually annotated traffic-vehicle data in Ultralytics YOLO format.

The preparation stage extracts representative frames, removes near-identical
consecutive samples, assigns whole source videos to dataset splits, and writes
manifests/guidance. It deliberately creates no annotation files and performs no
automatic labeling or model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "traffic_vehicles"
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("car", "motorcycle", "bus", "truck")


ANNOTATION_GUIDELINES = """# Traffic vehicle annotation requirements

## Classes

| ID | Class | Include |
|---:|---|---|
| 0 | car | Passenger cars, taxis, SUVs, pickups, and light passenger/cargo vans. |
| 1 | motorcycle | Motorcycles, scooters, and mopeds. Do not label the rider separately. |
| 2 | bus | City, school, shuttle, coach, and clearly passenger-service buses. |
| 3 | truck | Freight lorries, semitrailers, tankers, box trucks, dump trucks, and heavy commercial trucks. |

Use the definitions consistently. In this dataset, pickups and light vans are
`car`; large freight vehicles are `truck`.

## Bounding boxes

- Draw one tight box around every clearly identifiable target vehicle.
- Include the visible vehicle body and wheels, with as little road/background
  as practical.
- For a partially occluded vehicle, box the visible extent; do not guess the
  hidden boundary.
- Clip boxes at the image edge for truncated vehicles.
- Label small and distant vehicles when the class is still identifiable. Zoom
  in while annotating. Do not guess a class for an indistinguishable few-pixel
  object.
- Label each distinct vehicle separately in queues and overlaps.
- Do not label people, bicycles, trains, reflections, shadows, signs, or images
  of vehicles on billboards.
- Review dense frames at high zoom for missed objects.

## YOLO label format

Each image must have a matching `.txt` file under the corresponding `labels`
split. Each object is one line:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `[0, 1]`. Example:

```text
3 0.625000 0.540000 0.180000 0.220000
```

Create an empty `.txt` file only after confirming that an image contains no
target vehicles. A missing file means annotation is not complete.

## Coverage and quality review

- Include small/distant, near-camera, partially occluded, front/rear/side, and
  different-angle examples.
- Include varying illumination, weather, traffic density, road direction, and
  camera viewpoint.
- Seek additional source videos containing motorcycles, buses, and trucks;
  never duplicate frames merely to make class counts look balanced.
- Before training, verify every image has a reviewed label file and run a
  visual label audit on all validation and test images.
"""


@dataclass(frozen=True)
class VideoSource:
    """Validated source video and its assigned leakage-safe split."""

    path: Path
    split: str
    sha256: str


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation options."""
    parser = argparse.ArgumentParser(
        description="Extract diverse traffic frames for manual YOLO annotation."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video",
        action="append",
        help="Traffic video path; repeat for multiple distinct videos.",
    )
    source_group.add_argument(
        "--input-dir",
        help="Directory recursively scanned for supported video files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Dataset root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=0.5,
        help="Candidate sampling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--min-hash-distance",
        type=int,
        default=6,
        help="Minimum 64-bit dHash distance from the last kept frame (default: 6).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Extracted JPEG quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic video-level split seed (default: 42).",
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.70, 0.20, 0.10),
        help="Video-level split ratios (default: 0.70 0.20 0.10).",
    )
    args = parser.parse_args()

    if args.sample_every <= 0:
        parser.error("--sample-every must be greater than zero.")
    if not 0 <= args.min_hash_distance <= 64:
        parser.error("--min-hash-distance must be between 0 and 64.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100.")
    if any(value < 0 for value in args.split) or sum(args.split) <= 0:
        parser.error("--split values must be non-negative with a positive sum.")

    split_total = sum(args.split)
    args.split = tuple(value / split_total for value in args.split)
    return args


def discover_videos(args: argparse.Namespace) -> List[Path]:
    """Resolve explicitly selected or directory-discovered source videos."""
    if args.video:
        candidates = [Path(value).resolve() for value in args.video]
    else:
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.is_dir():
            raise ValueError(f"Input video directory does not exist: {input_dir}")
        candidates = sorted(
            path.resolve()
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    videos: List[Path] = []
    for path in candidates:
        if not path.is_file():
            raise ValueError(f"Video file does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {path}")
        videos.append(path)

    if not videos:
        raise ValueError("No supported source videos were found.")
    return videos


def file_sha256(path: Path) -> str:
    """Calculate a stable hash used to reject exact duplicate videos."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_exact_duplicate_videos(paths: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
    """Keep one copy of each exact video and return skipped duplicates."""
    seen: Dict[str, Path] = {}
    unique: List[Path] = []
    duplicates: List[Path] = []
    for path in paths:
        digest = file_sha256(path)
        if digest in seen:
            duplicates.append(path)
            continue
        seen[digest] = path
        unique.append(path)
    return unique, duplicates


def split_counts(video_count: int, ratios: Sequence[float]) -> List[int]:
    """Allocate whole videos while keeping all splits non-empty when possible."""
    if video_count == 1:
        return [1, 0, 0]
    if video_count == 2:
        return [1, 1, 0]

    counts = [1, 1, 1]
    for _ in range(video_count - 3):
        deficits = [ratios[i] * video_count - counts[i] for i in range(3)]
        counts[max(range(3), key=lambda i: deficits[i])] += 1
    return counts


def assign_video_splits(
    videos: Sequence[Path], ratios: Sequence[float], seed: int
) -> List[VideoSource]:
    """Assign complete source videos to train/val/test to avoid frame leakage."""
    shuffled = list(videos)
    random.Random(seed).shuffle(shuffled)
    counts = split_counts(len(shuffled), ratios)

    sources: List[VideoSource] = []
    offset = 0
    for split, count in zip(SPLITS, counts):
        for path in shuffled[offset : offset + count]:
            sources.append(VideoSource(path, split, file_sha256(path)))
        offset += count
    return sources


def difference_hash(frame: np.ndarray) -> int:
    """Return a 64-bit perceptual dHash for near-duplicate rejection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in comparisons.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    """Count differing bits between two perceptual hashes."""
    return (left ^ right).bit_count()


def create_structure(output_dir: Path) -> None:
    """Create a new empty Ultralytics dataset structure without overwriting data."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output dataset is not empty: {output_dir}. "
            "Choose a new directory to protect existing annotations."
        )
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def image_name(source: VideoSource, frame_index: int, timestamp: float) -> str:
    """Create a collision-resistant traceable extracted-image filename."""
    source_id = source.sha256[:10]
    milliseconds = round(timestamp * 1000)
    return f"{source.path.stem}_{source_id}_f{frame_index:07d}_t{milliseconds:010d}.jpg"


def extract_source(
    source: VideoSource,
    output_dir: Path,
    sample_every: float,
    min_hash_distance: int,
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    """Extract time-spaced, perceptually distinct frames from one source video."""
    capture = cv2.VideoCapture(str(source.path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {source.path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid FPS metadata: {source.path}")
    candidate_step = max(1, round(fps * sample_every))

    records: List[Dict[str, object]] = []
    last_kept_hash = None
    frame_index = 0
    candidates = 0
    rejected_similar = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % candidate_step != 0:
                frame_index += 1
                continue

            candidates += 1
            perceptual_hash = difference_hash(frame)
            distance = (
                64
                if last_kept_hash is None
                else hamming_distance(perceptual_hash, last_kept_hash)
            )
            if last_kept_hash is not None and distance < min_hash_distance:
                rejected_similar += 1
                frame_index += 1
                continue

            timestamp = frame_index / fps
            filename = image_name(source, frame_index, timestamp)
            relative_image = Path("images") / source.split / filename
            destination = output_dir / relative_image
            written = cv2.imwrite(
                str(destination),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not written:
                raise RuntimeError(f"Could not write extracted frame: {destination}")

            records.append(
                {
                    "image": relative_image.as_posix(),
                    "split": source.split,
                    "source_video": str(source.path),
                    "source_sha256": source.sha256,
                    "source_frame": frame_index,
                    "timestamp_seconds": round(timestamp, 3),
                    "dhash": f"{perceptual_hash:016x}",
                    "distance_from_previous_kept": distance,
                    "annotation_status": "pending",
                }
            )
            last_kept_hash = perceptual_hash
            frame_index += 1
    finally:
        capture.release()

    print(
        f"{source.path.name}: split={source.split}, candidates={candidates}, "
        f"kept={len(records)}, near-duplicates-rejected={rejected_similar}"
    )
    return records


def write_manifest(output_dir: Path, records: Sequence[Dict[str, object]]) -> None:
    """Write extraction provenance and annotation status."""
    path = output_dir / "manifest.csv"
    fieldnames = [
        "image",
        "split",
        "source_video",
        "source_sha256",
        "source_frame",
        "timestamp_seconds",
        "dhash",
        "distance_from_previous_kept",
        "annotation_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_sources_manifest(
    output_dir: Path,
    sources: Sequence[VideoSource],
    duplicates: Sequence[Path],
) -> None:
    """Record split assignments and exact duplicate files skipped."""
    path = output_dir / "sources.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["source_video", "split", "sha256", "status"])
        for source in sources:
            writer.writerow([source.path, source.split, source.sha256, "used"])
        for duplicate in duplicates:
            writer.writerow([duplicate, "", file_sha256(duplicate), "exact_duplicate_skipped"])


def write_dataset_yaml(output_dir: Path) -> None:
    """Create an Ultralytics-compatible dataset configuration."""
    yaml_text = f"""# Ultralytics YOLO traffic-vehicle dataset
path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    (output_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def print_summary(
    output_dir: Path,
    sources: Sequence[VideoSource],
    records: Sequence[Dict[str, object]],
    duplicates: Sequence[Path],
) -> None:
    """Print preparation totals and prominent safety warnings."""
    image_counts = Counter(record["split"] for record in records)
    video_counts = Counter(source.split for source in sources)
    print()
    print("Dataset preparation complete; annotations remain pending.")
    for split in SPLITS:
        print(
            f"{split}: {image_counts[split]} images from {video_counts[split]} videos"
        )
    print(f"Exact duplicate videos skipped: {len(duplicates)}")
    print(f"Dataset root: {output_dir}")
    print(f"Dataset YAML: {output_dir / 'dataset.yaml'}")
    print(f"Extraction manifest: {output_dir / 'manifest.csv'}")
    print("No label files were generated. Do not train until manual annotation and review are complete.")
    if any(video_counts[split] == 0 for split in ("val", "test")):
        print(
            "WARNING: Add more independent source videos before training so "
            "validation and test splits are non-empty and leakage-safe."
        )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        videos = discover_videos(args)
        unique_videos, duplicates = remove_exact_duplicate_videos(videos)
        sources = assign_video_splits(unique_videos, args.split, args.seed)

        output_dir = Path(args.output_dir).resolve()
        create_structure(output_dir)

        records: List[Dict[str, object]] = []
        for source in sources:
            records.extend(
                extract_source(
                    source,
                    output_dir,
                    sample_every=args.sample_every,
                    min_hash_distance=args.min_hash_distance,
                    jpeg_quality=args.jpeg_quality,
                )
            )

        write_manifest(output_dir, records)
        write_sources_manifest(output_dir, sources, duplicates)
        write_dataset_yaml(output_dir)
        (output_dir / "ANNOTATION_GUIDELINES.md").write_text(
            ANNOTATION_GUIDELINES,
            encoding="utf-8",
        )
        print_summary(output_dir, sources, records, duplicates)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
