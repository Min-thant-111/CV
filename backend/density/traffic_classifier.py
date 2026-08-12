"""Small frame classifier for TrafficDB light/medium/heavy congestion."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn


TRAFFIC_LEVELS = ("light", "medium", "heavy")
DEFAULT_INPUT_SIZE = (128, 96)  # width, height


class TrafficDensityCNN(nn.Module):
    """Compact CNN suitable for CPU inference on an edge device."""

    def __init__(self, class_count: int = len(TRAFFIC_LEVELS)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.classifier(features)


def frame_to_tensor(
    frame: np.ndarray, input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
) -> torch.Tensor:
    """Convert an OpenCV BGR frame into a normalized model tensor."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Invalid frame provided to traffic classifier.")
    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std


class TrafficDensityClassifier:
    """Load a trained classifier and predict a traffic level for one frame."""

    def __init__(self, model_path: str, device: str = "cpu"):
        checkpoint = torch.load(Path(model_path), map_location=device, weights_only=True)
        self.levels = tuple(checkpoint.get("class_names", TRAFFIC_LEVELS))
        self.input_size = tuple(checkpoint.get("input_size", DEFAULT_INPUT_SIZE))
        self.device = torch.device(device)
        self.model = TrafficDensityCNN(len(self.levels)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        image = frame_to_tensor(frame, self.input_size).unsqueeze(0).to(self.device)
        probabilities = self.model(image).softmax(dim=1)[0]
        index = int(probabilities.argmax().item())
        return self.levels[index], round(float(probabilities[index].item()), 4)
