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
    single_flock: bool = False,
) -> dict | None:
    if not candidates:
        return None

    if not single_flock and active_track_id is not None:
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
    flock_center_x = float(current_row["flock_center_x"])
    flock_center_y = float(current_row["flock_center_y"])
    major_axis, minor_axis, ellipse_angle = get_ellipse_parameters(current_row)

    center_x_norm = flock_center_x / frame_width
    center_y_norm = flock_center_y / frame_height
    major_axis_norm = major_axis / frame_width
    minor_axis_norm = minor_axis / frame_height
    aspect_ratio = major_axis / minor_axis
    area_ratio = (math.pi * major_axis * minor_axis) / (frame_width * frame_height)
    angle_radians = math.radians(ellipse_angle)
    angle_cos = math.cos(angle_radians)
    angle_sin = math.sin(angle_radians)

    velocity_x = 0.0
    velocity_y = 0.0
    major_axis_change = 0.0
    minor_axis_change = 0.0

    if previous_row is not None:
        previous_major_axis, previous_minor_axis, _ = get_ellipse_parameters(previous_row)
        position_scale = 0.5 * (major_axis + previous_major_axis)
        minor_scale = 0.5 * (minor_axis + previous_minor_axis)

        velocity_x = (
            flock_center_x - float(previous_row["flock_center_x"])
        ) / max(1.0, position_scale)
        velocity_y = (
            flock_center_y - float(previous_row["flock_center_y"])
        ) / max(1.0, position_scale)
        major_axis_change = (major_axis - previous_major_axis) / max(1.0, position_scale)
        minor_axis_change = (minor_axis - previous_minor_axis) / max(1.0, minor_scale)

    return np.array(
        [
            center_x_norm,
            center_y_norm,
            major_axis_norm,
            minor_axis_norm,
            aspect_ratio,
            area_ratio,
            angle_cos,
            angle_sin,
            velocity_x,
            velocity_y,
            major_axis_change,
            minor_axis_change,
        ],
        dtype=np.float32,
    )


def get_ellipse_parameters(row: dict) -> tuple[float, float, float]:
    major_axis = row.get("flock_ellipse_major_axis", "")
    minor_axis = row.get("flock_ellipse_minor_axis", "")
    angle = row.get("flock_ellipse_angle", "")

    if major_axis != "" and minor_axis != "" and angle != "":
        return max(1.0, float(major_axis)), max(1.0, float(minor_axis)), float(angle)

    if "ellipse_major_axis" in row and "ellipse_minor_axis" in row:
        return (
            max(1.0, float(row["ellipse_major_axis"])),
            max(1.0, float(row["ellipse_minor_axis"])),
            float(row.get("ellipse_angle", 0.0) or 0.0),
        )

    flock_width = max(1.0, float(row["flock_width"]))
    flock_height = max(1.0, float(row["flock_height"]))
    return max(flock_width, flock_height) / 2.0, min(flock_width, flock_height) / 2.0, 0.0


def calculate_rear_direction(
    current_row: dict,
    previous_row: dict | None = None,
) -> tuple[float, float]:
    row_rear_x = current_row.get("rear_direction_x", "")
    row_rear_y = current_row.get("rear_direction_y", "")

    if row_rear_x != "" and row_rear_y != "":
        rear_x = float(row_rear_x)
        rear_y = float(row_rear_y)
        magnitude = math.hypot(rear_x, rear_y)

        if magnitude > 1e-6:
            return rear_x / magnitude, rear_y / magnitude

    if previous_row is not None:
        dx = (
            float(current_row["flock_center_x"])
            - float(previous_row["flock_center_x"])
        )
        dy = (
            float(current_row["flock_center_y"])
            - float(previous_row["flock_center_y"])
        )
        magnitude = math.hypot(dx, dy)

        if magnitude > 1e-6:
            return -dx / magnitude, -dy / magnitude

    # En coordenadas de imagen, Y positiva apunta hacia abajo.
    return 0.0, 1.0


def calculate_stable_rear_direction(
    trajectory: deque[tuple[int, int]],
    last_rear_direction: tuple[float, float] | None,
    window: int = 12,
    dead_zone: float = 12.0,
    smoothing: float = 0.18,
) -> tuple[tuple[float, float] | None, float, str]:
    """
    Estima una dirección trasera robusta desde una ventana de centros.

    Si el movimiento no es fiable, conserva la última dirección válida.
    """

    if len(trajectory) < 2:
        if last_rear_direction is not None:
            return last_rear_direction, 0.0, "FROZEN"

        return None, 0.0, "UNKNOWN"

    effective_window = min(window, len(trajectory) - 1)
    points = list(trajectory)[-effective_window - 1:]
    time = np.arange(len(points), dtype=float)
    x_values = np.array([point[0] for point in points], dtype=float)
    y_values = np.array([point[1] for point in points], dtype=float)

    time_centered = time - float(time.mean())
    denominator = float(np.dot(time_centered, time_centered))

    if denominator < 1e-6:
        if last_rear_direction is not None:
            return last_rear_direction, 0.0, "FROZEN"

        return None, 0.0, "UNKNOWN"

    slope_x = float(np.dot(time_centered, x_values - float(x_values.mean())) / denominator)
    slope_y = float(np.dot(time_centered, y_values - float(y_values.mean())) / denominator)
    dx = slope_x * effective_window
    dy = slope_y * effective_window
    magnitude = math.hypot(dx, dy)

    if magnitude < dead_zone:
        if last_rear_direction is not None:
            return last_rear_direction, 0.0, "FROZEN"

        return None, 0.0, "UNKNOWN"

    raw_rear_x = -dx / magnitude
    raw_rear_y = -dy / magnitude
    residuals = []

    for index, point in enumerate(points):
        centered_index = index - float(time.mean())
        predicted_x = float(x_values.mean()) + slope_x * centered_index
        predicted_y = float(y_values.mean()) + slope_y * centered_index
        residuals.append(math.hypot(point[0] - predicted_x, point[1] - predicted_y))

    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    fit_quality = 1.0 / (1.0 + mean_residual / max(magnitude, 1.0))
    motion_strength = min(1.0, magnitude / max(dead_zone * 4.0, 1.0))
    confidence = motion_strength * fit_quality

    if confidence < 0.20:
        if last_rear_direction is not None:
            return last_rear_direction, confidence, "FROZEN"

        return None, confidence, "UNKNOWN"

    if last_rear_direction is None:
        return (raw_rear_x, raw_rear_y), confidence, "MOTION"

    mixed_x = (1.0 - smoothing) * last_rear_direction[0] + smoothing * raw_rear_x
    mixed_y = (1.0 - smoothing) * last_rear_direction[1] + smoothing * raw_rear_y
    mixed_magnitude = math.hypot(mixed_x, mixed_y)

    if mixed_magnitude < 1e-6:
        return last_rear_direction, 0.0, "FROZEN"

    return (
        (mixed_x / mixed_magnitude, mixed_y / mixed_magnitude),
        confidence,
        "MOTION",
    )


def enrich_row_with_rear_direction(
    current_row: dict,
    previous_row: dict | None = None,
) -> dict:
    rear_x, rear_y = calculate_rear_direction(current_row, previous_row)
    enriched_row = current_row.copy()
    enriched_row["rear_direction_x"] = rear_x
    enriched_row["rear_direction_y"] = rear_y
    enriched_row["lateral_direction_x"] = -rear_y
    enriched_row["lateral_direction_y"] = rear_x
    return enriched_row


def build_target_vector(
    current_row: dict,
    future_row: dict,
    previous_row: dict | None = None,
) -> np.ndarray:
    current_row = enrich_row_with_rear_direction(current_row, previous_row)
    rear_scale, lateral_scale, _ = get_ellipse_parameters(current_row)

    rear_x = float(current_row["rear_direction_x"])
    rear_y = float(current_row["rear_direction_y"])
    lateral_x = float(current_row["lateral_direction_x"])
    lateral_y = float(current_row["lateral_direction_y"])

    target_vector_x = (
        float(future_row["dog_center_x"])
        - float(current_row["flock_center_x"])
    )

    target_vector_y = (
        float(future_row["dog_center_y"])
        - float(current_row["flock_center_y"])
    )

    rear_distance = (
        target_vector_x * rear_x
        + target_vector_y * rear_y
    ) / rear_scale

    lateral_offset = (
        target_vector_x * lateral_x
        + target_vector_y * lateral_y
    ) / lateral_scale

    return np.array(
        [rear_distance, lateral_offset],
        dtype=np.float32,
    )


def denormalize_target(
    current_row: dict,
    prediction: np.ndarray,
) -> tuple[float, float]:
    current_row = enrich_row_with_rear_direction(current_row)
    rear_scale, lateral_scale, _ = get_ellipse_parameters(current_row)
    rear_distance = max(0.15, float(prediction[0]))
    lateral_offset = max(-1.25, min(1.25, float(prediction[1])))

    rear_x = float(current_row["rear_direction_x"])
    rear_y = float(current_row["rear_direction_y"])
    lateral_x = float(current_row["lateral_direction_x"])
    lateral_y = float(current_row["lateral_direction_y"])

    predicted_x = float(current_row["flock_center_x"]) + (
        rear_distance * rear_scale * rear_x
        + lateral_offset * lateral_scale * lateral_x
    )
    predicted_y = float(current_row["flock_center_y"]) + (
        rear_distance * rear_scale * rear_y
        + lateral_offset * lateral_scale * lateral_y
    )
    
    return predicted_x, predicted_y


def constrain_point_to_rear(
    current_row: dict,
    point: tuple[float, float],
    min_rear_distance: float = 0.15,
    max_lateral_offset: float = 1.25,
) -> tuple[float, float]:
    current_row = enrich_row_with_rear_direction(current_row)
    rear_scale, lateral_scale, _ = get_ellipse_parameters(current_row)
    rear_x = float(current_row["rear_direction_x"])
    rear_y = float(current_row["rear_direction_y"])
    lateral_x = float(current_row["lateral_direction_x"])
    lateral_y = float(current_row["lateral_direction_y"])

    vector_x = point[0] - float(current_row["flock_center_x"])
    vector_y = point[1] - float(current_row["flock_center_y"])
    rear_distance = max(
        min_rear_distance,
        (vector_x * rear_x + vector_y * rear_y) / rear_scale,
    )
    lateral_offset = max(
        -max_lateral_offset,
        min(
            max_lateral_offset,
            (vector_x * lateral_x + vector_y * lateral_y) / lateral_scale,
        ),
    )

    constrained_x = float(current_row["flock_center_x"]) + (
        rear_distance * rear_scale * rear_x
        + lateral_offset * lateral_scale * lateral_x
    )
    constrained_y = float(current_row["flock_center_y"]) + (
        rear_distance * rear_scale * rear_y
        + lateral_offset * lateral_scale * lateral_y
    )
    return constrained_x, constrained_y


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
    major_axis, _, _ = get_ellipse_parameters(current_flock)
    offset = 0.9 * major_axis

    return (
        float(current_flock["flock_center_x"]) + offset * unit_back_x,
        float(current_flock["flock_center_y"]) + offset * unit_back_y,
    )
