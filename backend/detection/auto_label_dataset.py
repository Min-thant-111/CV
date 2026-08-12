"""Create reviewable YOLO labels for extracted traffic frames.

The labels produced here are pseudo-labels, not ground truth. They are intended
to accelerate a human annotation pass before fine-tuning a detector.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from ultralytics import YOLO


COCO_TO_TRAFFIC = {2: 0, 3: 1, 5: 2, 7: 3}
SPLITS = ("train", "val", "test")


def batched(items: List[Path], size: int) -> Iterable[List[Path]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate draft vehicle labels for an extracted YOLO dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    parser.add_argument("--model", default="yolov8n.pt", help="Teacher YOLO weights.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing label files."
    )
    args = parser.parse_args()
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0, 1].")
    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be in (0, 1].")
    if args.batch < 1:
        parser.error("--batch must be positive.")
    return args


def update_manifest(dataset: Path) -> None:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        return
    with manifest.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    for row in rows:
        image = dataset / row["image"]
        label = dataset / "labels" / row["split"] / f"{image.stem}.txt"
        if label.is_file():
            row["annotation_status"] = "auto_labeled_pending_review"
    with manifest.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset}")

    model = YOLO(args.model)
    totals = Counter()
    class_totals = Counter()

    for split in SPLITS:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (args.overwrite or not (label_dir / f"{path.stem}.txt").exists())
        )

        for image_batch in batched(images, args.batch):
            results = model.predict(
                source=[str(path) for path in image_batch],
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=list(COCO_TO_TRAFFIC),
                device=args.device,
                verbose=False,
            )
            for image, result in zip(image_batch, results):
                lines = []
                boxes = result.boxes
                if boxes is not None:
                    for class_id, xywhn in zip(
                        boxes.cls.int().cpu().tolist(),
                        boxes.xywhn.cpu().tolist(),
                    ):
                        mapped_id = COCO_TO_TRAFFIC.get(class_id)
                        if mapped_id is None:
                            continue
                        x_center, y_center, width, height = xywhn
                        lines.append(
                            f"{mapped_id} {x_center:.6f} {y_center:.6f} "
                            f"{width:.6f} {height:.6f}"
                        )
                        class_totals[mapped_id] += 1
                (label_dir / f"{image.stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                totals[split] += 1
        print(f"{split}: auto-labeled {totals[split]} images")

    update_manifest(dataset)
    print(f"Images labeled: {sum(totals.values())}")
    print(
        "Objects: "
        f"car={class_totals[0]}, motorcycle={class_totals[1]}, "
        f"bus={class_totals[2]}, truck={class_totals[3]}"
    )
    print("Status: auto_labeled_pending_review (human review required before training)")


if __name__ == "__main__":
    main()
