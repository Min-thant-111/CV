"""Train the TrafficDB frame-level congestion classifier."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.density.traffic_classifier import (
    DEFAULT_INPUT_SIZE,
    TRAFFIC_LEVELS,
    TrafficDensityCNN,
    frame_to_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on official TrafficDB labels.")
    parser.add_argument("--dataset", required=True, help="Extracted dataset root.")
    parser.add_argument("--info", required=True, help="TrafficDB info.txt path.")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "models" / "traffic_density_cnn.pt")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_video_labels(path: Path) -> Dict[str, str]:
    labels = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.startswith("#"):
            continue
        fields = raw_line.split()
        labels[fields[0]] = fields[8]
    return labels


def read_samples(dataset: Path, info: Path) -> Dict[str, List[Tuple[Path, int]]]:
    video_labels = read_video_labels(info)
    level_ids = {name: index for index, name in enumerate(TRAFFIC_LEVELS)}
    samples = {split: [] for split in ("train", "val", "test")}
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            source_name = Path(row["source_video"]).stem
            level = video_labels.get(source_name)
            if level not in level_ids:
                raise ValueError(f"No valid TrafficDB label for {source_name}")
            samples[row["split"]].append(
                (dataset / row["image"], level_ids[level])
            )
    return samples


class TrafficFrames(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read training image: {path}")
        if self.augment:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-8.0, 8.0)
            frame = cv2.convertScaleAbs(frame, alpha=gain, beta=bias)
        return frame_to_tensor(frame), label


@torch.inference_mode()
def evaluate(model, loader, device) -> Tuple[float, List[List[int]]]:
    model.eval()
    correct = 0
    total = 0
    confusion = [[0] * len(TRAFFIC_LEVELS) for _ in TRAFFIC_LEVELS]
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == labels).sum().item())
        total += len(labels)
        for truth, prediction in zip(labels.tolist(), predictions.tolist()):
            confusion[truth][prediction] += 1
    return correct / max(1, total), confusion


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Path(args.dataset).resolve()
    samples = read_samples(dataset, Path(args.info).resolve())
    device = torch.device(args.device)

    counts = Counter(label for _, label in samples["train"])
    weights = torch.tensor(
        [len(samples["train"]) / (len(TRAFFIC_LEVELS) * counts[i]) for i in range(3)],
        dtype=torch.float32,
        device=device,
    )
    loaders = {
        split: DataLoader(
            TrafficFrames(rows, augment=split == "train"),
            batch_size=args.batch,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, rows in samples.items()
    }
    model = TrafficDensityCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weights)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for images, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(labels)
        validation_accuracy, _ = evaluate(model, loaders["val"], device)
        print(
            f"epoch={epoch:02d} loss={loss_total / len(samples['train']):.4f} "
            f"val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": TRAFFIC_LEVELS,
                    "input_size": DEFAULT_INPUT_SIZE,
                    "validation_accuracy": best_accuracy,
                },
                output,
            )

    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_accuracy, confusion = evaluate(model, loaders["test"], device)
    print(f"best_validation_accuracy={best_accuracy:.4f}")
    print(f"test_accuracy={test_accuracy:.4f}")
    print("confusion_rows=true_columns=predicted")
    for level, row in zip(TRAFFIC_LEVELS, confusion):
        print(f"{level}: {row}")
    print(f"model={output}")


if __name__ == "__main__":
    main()
