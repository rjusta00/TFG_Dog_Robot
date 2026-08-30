import csv
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
from ultralytics import YOLO


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calculate_robot_guidance import clip_point
from simulate_robot_mpc import resolve_project_path
from track_flock_motion import calculate_box_center


DEFAULT_DETECTOR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dogRobot_v2_best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "flock_only_target_prediction"
)


def select_main_flock(
    candidates: list[dict],
    active_track_id: int | None,
) -> dict | None:
    if not candidates:
        return None

    if active_track_id is not None:
        for candidate in candidates:
            if candidate["track_id"] == active_track_id:
                return candidate

    return max(candidates, key=lambda candidate: candidate["area"])


def extract_flock_candidates(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = None

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    candidates: list[dict] = []

    for index, box in enumerate(boxes_xyxy):
        if int(class_ids[index]) != 0:
            continue

        x1, y1, x2, y2 = box.tolist()
        center = calculate_box_center(x1, y1, x2, y2)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "center": center,
                "confidence": float(confidences[index]),
                "track_id": (
                    None
                    if track_ids is None
                    else int(track_ids[index])
                ),
                "width": width,
                "height": height,
                "area": width * height,
            }
        )

    return candidates


def build_flock_feature_vector(
    current_row: dict,
    previous_row: dict | None,
) -> np.ndarray:
    frame_width = max(1.0, float(current_row["frame_width"]))
    frame_height = max(1.0, float(current_row["frame_height"]))
    flock_width = max(1.0, float(current_row["flock_width"]))
    flock_height = max(1.0, float(current_row["flock_height"]))
    flock_center_x = float(current_row["flock_center_x"])
    flock_center_y = float(current_row["flock_center_y"])

    center_x_norm = flock_center_x / frame_width
    center_y_norm = flock_center_y / frame_height
    width_norm = flock_width / frame_width
    height_norm = flock_height / frame_height
    aspect_ratio = flock_width / flock_height
    area_ratio = (flock_width * flock_height) / (frame_width * frame_height)

    velocity_x = 0.0
    velocity_y = 0.0
    scale_change_x = 0.0
    scale_change_y = 0.0

    if previous_row is not None:
        previous_flock_width = max(1.0, float(previous_row["flock_width"]))
        previous_flock_height = max(1.0, float(previous_row["flock_height"]))
        scale_x = 0.5 * (flock_width + previous_flock_width)
        scale_y = 0.5 * (flock_height + previous_flock_height)

        velocity_x = (
            flock_center_x - float(previous_row["flock_center_x"])
        ) / scale_x
        velocity_y = (
            flock_center_y - float(previous_row["flock_center_y"])
        ) / scale_y
        scale_change_x = (flock_width - previous_flock_width) / scale_x
        scale_change_y = (flock_height - previous_flock_height) / scale_y

    return np.array(
        [
            center_x_norm,
            center_y_norm,
            width_norm,
            height_norm,
            aspect_ratio,
            area_ratio,
            velocity_x,
            velocity_y,
            scale_change_x,
            scale_change_y,
        ],
        dtype=np.float32,
    )


def build_target_vector(
    current_row: dict,
    future_row: dict,
) -> np.ndarray:
    flock_width = max(1.0, float(current_row["flock_width"]))
    flock_height = max(1.0, float(current_row["flock_height"]))

    relative_target_x = (
        float(future_row["dog_center_x"])
        - float(current_row["flock_center_x"])
    ) / flock_width

    relative_target_y = (
        float(future_row["dog_center_y"])
        - float(current_row["flock_center_y"])
    ) / flock_height

    return np.array(
        [relative_target_x, relative_target_y],
        dtype=np.float32,
    )


def denormalize_target(
    current_row: dict,
    prediction: np.ndarray,
) -> tuple[float, float]:
    flock_width = max(1.0, float(current_row["flock_width"]))
    flock_height = max(1.0, float(current_row["flock_height"]))

    predicted_x = float(current_row["flock_center_x"]) + float(prediction[0]) * flock_width
    predicted_y = float(current_row["flock_center_y"]) + float(prediction[1]) * flock_height
    return predicted_x, predicted_y


def load_rows(dataset_csv: Path) -> list[dict]:
    with dataset_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def group_rows_by_video(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}

    for row in rows:
        grouped.setdefault(row["video_id"], []).append(row)

    for video_id in grouped:
        grouped[video_id].sort(key=lambda row: int(row["frame"]))

    return grouped


def load_detector(model_path: Path) -> YOLO:
    return YOLO(str(resolve_project_path(model_path)))


def build_linear_target_from_flock(
    flock_history: deque[dict],
    target_history: deque[tuple[int, int]],
) -> tuple[float, float] | None:
    if len(flock_history) < 2 or len(target_history) < 2:
        return None

    flock_last = flock_history[-1]
    flock_prev = flock_history[-2]
    dx = float(flock_last["flock_center_x"]) - float(flock_prev["flock_center_x"])
    dy = float(flock_last["flock_center_y"]) - float(flock_prev["flock_center_y"])

    last_target_x, last_target_y = target_history[-1]
    return last_target_x + dx, last_target_y + dy


def estimate_target_from_flock_motion(
    flock_history: deque[dict],
) -> tuple[float, float] | None:
    if len(flock_history) < 3:
        return None

    current_flock = flock_history[-1]
    reference_index = max(0, len(flock_history) - 4)
    previous_flock = flock_history[reference_index]

    dx = float(current_flock["flock_center_x"]) - float(previous_flock["flock_center_x"])
    dy = float(current_flock["flock_center_y"]) - float(previous_flock["flock_center_y"])
    magnitude = math.hypot(dx, dy)

    if magnitude < 1e-6:
        return None

    unit_back_x = -dx / magnitude
    unit_back_y = -dy / magnitude
    flock_width = max(1.0, float(current_flock["flock_width"]))
    flock_height = max(1.0, float(current_flock["flock_height"]))
    offset = 0.45 * max(flock_width, flock_height)

    return (
        float(current_flock["flock_center_x"]) + offset * unit_back_x,
        float(current_flock["flock_center_y"]) + offset * unit_back_y,
    )
