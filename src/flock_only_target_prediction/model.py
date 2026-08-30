import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flock_only_target_prediction.common import (
    build_flock_feature_vector,
    build_target_vector,
)


@dataclass
class WindowMetadata:
    video_id: str
    current_row: dict
    future_row: dict


class TemporalWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class FlockTargetGRUPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(features)
        return self.head(outputs[:, -1, :])


def build_windows(
    rows_by_video: dict[str, list[dict]],
    history_length: int,
    prediction_offset_frames: int,
    allowed_videos: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[WindowMetadata]]:
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_metadata: list[WindowMetadata] = []

    for video_id, video_rows in rows_by_video.items():
        if allowed_videos is not None and video_id not in allowed_videos:
            continue

        segments: dict[str, list[dict]] = {}

        for row in video_rows:
            segment_id = row.get("segment_id", "")
            if segment_id == "":
                continue
            segments.setdefault(f"{video_id}:{segment_id}", []).append(row)

        for segment_rows in segments.values():
            segment_rows.sort(key=lambda row: int(row["frame"]))

            if len(segment_rows) < history_length + prediction_offset_frames:
                continue

            segment_features: list[np.ndarray] = []
            previous_row = None

            for row in segment_rows:
                if previous_row is not None and int(row["frame"]) != int(previous_row["frame"]) + 1:
                    previous_row = None

                segment_features.append(
                    build_flock_feature_vector(row, previous_row)
                )
                previous_row = row

            for end_index in range(history_length - 1, len(segment_rows) - prediction_offset_frames):
                start_index = end_index - history_length + 1
                future_index = end_index + prediction_offset_frames
                history_rows = segment_rows[start_index:end_index + 1]
                future_row = segment_rows[future_index]

                contiguous = True

                for index in range(1, len(history_rows)):
                    if int(history_rows[index]["frame"]) != int(history_rows[index - 1]["frame"]) + 1:
                        contiguous = False
                        break

                if not contiguous:
                    continue

                current_row = segment_rows[end_index]
                if int(future_row["frame"]) != int(current_row["frame"]) + prediction_offset_frames:
                    continue

                feature_window = np.stack(segment_features[start_index:end_index + 1], axis=0)
                target = build_target_vector(current_row, future_row)

                all_features.append(feature_window)
                all_targets.append(target)
                all_metadata.append(WindowMetadata(video_id, current_row, future_row))

    if not all_features:
        return (
            np.empty((0, history_length, 10), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            [],
        )

    return np.stack(all_features).astype(np.float32), np.stack(all_targets).astype(np.float32), all_metadata


def normalize_features(features: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.size == 0:
        raise ValueError("Cannot normalize an empty feature array.")

    if mean is None:
        mean = features.mean(axis=(0, 1), keepdims=True)

    if std is None:
        std = features.std(axis=(0, 1), keepdims=True)

    std = np.where(std < 1e-6, 1.0, std)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)
