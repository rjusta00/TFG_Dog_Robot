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


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dogRobot_v2_best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "rear_dog_prediction"
)


def calculate_box_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int]:
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    return center_x, center_y


def extract_tracked_candidates(result) -> tuple[list[dict], list[dict]]:
    if result.boxes is None or len(result.boxes) == 0:
        return [], []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    track_ids = None

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    flocks: list[dict] = []
    dogs: list[dict] = []

    for index, box in enumerate(boxes_xyxy):
        class_id = int(class_ids[index])

        if class_id not in {0, 1}:
            continue

        x1, y1, x2, y2 = box.tolist()
        center = calculate_box_center(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        area = width * height

        track_id = None

        if track_ids is not None:
            track_id = int(track_ids[index])

        candidate = {
            "box": (x1, y1, x2, y2),
            "center": center,
            "confidence": float(confidences[index]),
            "track_id": track_id,
            "area": area,
            "width": width,
            "height": height,
        }

        if class_id == 0:
            flocks.append(candidate)
        else:
            dogs.append(candidate)

    return flocks, dogs


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

    return max(
        candidates,
        key=lambda candidate: candidate["area"],
    )


def calculate_motion_vector(
    trajectory: deque[tuple[int, int]],
    window: int,
    dead_zone: float,
) -> tuple[np.ndarray | None, float]:
    if len(trajectory) < 2:
        return None, 0.0

    effective_window = min(
        window,
        len(trajectory) - 1,
    )

    start_x, start_y = trajectory[-effective_window - 1]
    end_x, end_y = trajectory[-1]

    vector = np.array(
        [
            float(end_x - start_x),
            float(end_y - start_y),
        ],
        dtype=float,
    )

    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude < dead_zone:
        return None, magnitude

    return vector / magnitude, magnitude


class RearDogSelector:
    def __init__(
        self,
        min_projection: float,
        lateral_penalty: float,
    ) -> None:
        self.active_track_id: int | None = None
        self.min_projection = min_projection
        self.lateral_penalty = lateral_penalty

    def clear_active_track(self) -> None:
        self.active_track_id = None

    def select(
        self,
        dog_candidates: list[dict],
        flock_center: tuple[int, int],
        rear_direction: np.ndarray | None,
    ) -> dict | None:
        if not dog_candidates or rear_direction is None:
            return None

        behind_candidates: list[dict] = []

        for candidate in dog_candidates:
            dog_x, dog_y = candidate["center"]
            vector = np.array(
                [
                    float(dog_x - flock_center[0]),
                    float(dog_y - flock_center[1]),
                ],
                dtype=float,
            )

            projection = float(
                np.dot(vector, rear_direction)
            )

            lateral_vector = vector - projection * rear_direction
            lateral_distance = float(
                np.linalg.norm(lateral_vector)
            )

            if projection < self.min_projection:
                continue

            score = projection - self.lateral_penalty * lateral_distance

            enriched_candidate = candidate.copy()
            enriched_candidate["rear_projection"] = projection
            enriched_candidate["lateral_distance"] = lateral_distance
            enriched_candidate["selection_score"] = score
            behind_candidates.append(enriched_candidate)

        if not behind_candidates:
            return None

        if self.active_track_id is not None:
            for candidate in behind_candidates:
                if candidate["track_id"] == self.active_track_id:
                    return candidate

        selected = max(
            behind_candidates,
            key=lambda candidate: candidate["selection_score"],
        )

        self.active_track_id = selected["track_id"]
        return selected


def build_frame_record(
    video_id: str,
    video_path: Path,
    frame_index: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    flock_candidate: dict,
    dog_candidate: dict,
    segment_id: int,
) -> dict[str, str | int | float]:
    flock_x1, flock_y1, flock_x2, flock_y2 = flock_candidate["box"]
    dog_x1, dog_y1, dog_x2, dog_y2 = dog_candidate["box"]
    flock_center_x, flock_center_y = flock_candidate["center"]
    dog_center_x, dog_center_y = dog_candidate["center"]

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "frame": frame_index,
        "time_seconds": frame_index / fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "segment_id": segment_id,
        "flock_track_id": (
            ""
            if flock_candidate["track_id"] is None
            else flock_candidate["track_id"]
        ),
        "dog_track_id": (
            ""
            if dog_candidate["track_id"] is None
            else dog_candidate["track_id"]
        ),
        "flock_center_x": flock_center_x,
        "flock_center_y": flock_center_y,
        "dog_center_x": dog_center_x,
        "dog_center_y": dog_center_y,
        "flock_box_x1": flock_x1,
        "flock_box_y1": flock_y1,
        "flock_box_x2": flock_x2,
        "flock_box_y2": flock_y2,
        "dog_box_x1": dog_x1,
        "dog_box_y1": dog_y1,
        "dog_box_x2": dog_x2,
        "dog_box_y2": dog_y2,
        "flock_width": flock_candidate["width"],
        "flock_height": flock_candidate["height"],
        "dog_width": dog_candidate["width"],
        "dog_height": dog_candidate["height"],
        "flock_confidence": flock_candidate["confidence"],
        "dog_confidence": dog_candidate["confidence"],
        "rear_projection": dog_candidate["rear_projection"],
        "lateral_distance": dog_candidate["lateral_distance"],
        "selection_score": dog_candidate["selection_score"],
    }


def build_feature_vector(
    current_row: dict,
    previous_row: dict | None,
) -> np.ndarray:
    flock_width = max(
        1.0,
        float(current_row["flock_width"]),
    )
    flock_height = max(
        1.0,
        float(current_row["flock_height"]),
    )
    frame_width = max(
        1.0,
        float(current_row["frame_width"]),
    )
    frame_height = max(
        1.0,
        float(current_row["frame_height"]),
    )

    dog_center_x = float(current_row["dog_center_x"])
    dog_center_y = float(current_row["dog_center_y"])
    flock_center_x = float(current_row["flock_center_x"])
    flock_center_y = float(current_row["flock_center_y"])

    relative_x = (
        dog_center_x - flock_center_x
    ) / flock_width
    relative_y = (
        dog_center_y - flock_center_y
    ) / flock_height

    dog_velocity_x = 0.0
    dog_velocity_y = 0.0
    flock_velocity_x = 0.0
    flock_velocity_y = 0.0

    if previous_row is not None:
        previous_flock_width = max(
            1.0,
            float(previous_row["flock_width"]),
        )
        previous_flock_height = max(
            1.0,
            float(previous_row["flock_height"]),
        )

        scale_x = 0.5 * (
            flock_width + previous_flock_width
        )

        scale_y = 0.5 * (
            flock_height + previous_flock_height
        )

        dog_velocity_x = (
            dog_center_x
            - float(previous_row["dog_center_x"])
        ) / scale_x

        dog_velocity_y = (
            dog_center_y
            - float(previous_row["dog_center_y"])
        ) / scale_y

        flock_velocity_x = (
            flock_center_x
            - float(previous_row["flock_center_x"])
        ) / scale_x

        flock_velocity_y = (
            flock_center_y
            - float(previous_row["flock_center_y"])
        ) / scale_y

    dog_width_ratio = float(current_row["dog_width"]) / flock_width
    dog_height_ratio = float(current_row["dog_height"]) / flock_height
    flock_width_ratio = flock_width / frame_width
    flock_height_ratio = flock_height / frame_height

    return np.array(
        [
            relative_x,
            relative_y,
            dog_velocity_x,
            dog_velocity_y,
            flock_velocity_x,
            flock_velocity_y,
            dog_width_ratio,
            dog_height_ratio,
            flock_width_ratio,
            flock_height_ratio,
        ],
        dtype=np.float32,
    )


def build_target_vector(
    current_row: dict,
    future_row: dict,
) -> np.ndarray:
    flock_width = max(
        1.0,
        float(current_row["flock_width"]),
    )
    flock_height = max(
        1.0,
        float(current_row["flock_height"]),
    )

    target_x = (
        float(future_row["dog_center_x"])
        - float(current_row["flock_center_x"])
    ) / flock_width

    target_y = (
        float(future_row["dog_center_y"])
        - float(current_row["flock_center_y"])
    ) / flock_height

    return np.array(
        [target_x, target_y],
        dtype=np.float32,
    )


def denormalize_prediction(
    current_row: dict,
    prediction: np.ndarray,
) -> tuple[float, float]:
    flock_width = max(
        1.0,
        float(current_row["flock_width"]),
    )
    flock_height = max(
        1.0,
        float(current_row["flock_height"]),
    )

    predicted_x = (
        float(current_row["flock_center_x"])
        + float(prediction[0]) * flock_width
    )

    predicted_y = (
        float(current_row["flock_center_y"])
        + float(prediction[1]) * flock_height
    )

    return predicted_x, predicted_y


def load_tracking_rows(
    dataset_csv: Path,
) -> list[dict]:
    with dataset_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def group_rows_by_video(
    rows: list[dict],
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}

    for row in rows:
        grouped.setdefault(
            row["video_id"],
            [],
        ).append(row)

    for video_id in grouped:
        grouped[video_id].sort(
            key=lambda row: int(row["frame"])
        )

    return grouped


def load_yolo_model(model_path: Path) -> YOLO:
    return YOLO(str(resolve_project_path(model_path)))
