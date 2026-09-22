import argparse
import csv
import json
import math
import random
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flock_only_target_prediction.common import (
    DEFAULT_DETECTOR_MODEL_PATH,
    OUTPUT_ROOT,
    build_target_vector,
    calculate_stable_rear_direction,
    denormalize_target,
    group_rows_by_video,
    load_detector,
    load_rows,
    resolve_project_path,
    select_main_flock,
)
from flock_only_target_prediction.model import (
    FlockTargetGRUPredictor,
    TemporalWindowDataset,
    build_windows,
    normalize_features,
)
from track_flock_motion import calculate_flock_ellipse

MODELS_ROOT = PROJECT_ROOT / "models"

DATASET_FIELDNAMES = [
    "video_id",
    "video_path",
    "frame",
    "time_seconds",
    "frame_width",
    "frame_height",
    "segment_id",
    "flock_track_id",
    "dog_track_id",
    "flock_center_x",
    "flock_center_y",
    "dog_center_x",
    "dog_center_y",
    "flock_box_x1",
    "flock_box_y1",
    "flock_box_x2",
    "flock_box_y2",
    "flock_ellipse_center_x",
    "flock_ellipse_center_y",
    "flock_ellipse_major_axis",
    "flock_ellipse_minor_axis",
    "flock_ellipse_angle",
    "dog_box_x1",
    "dog_box_y1",
    "dog_box_x2",
    "dog_box_y2",
    "flock_width",
    "flock_height",
    "dog_width",
    "dog_height",
    "flock_confidence",
    "dog_confidence",
    "rear_direction_x",
    "rear_direction_y",
    "rear_direction_confidence",
    "rear_direction_source",
    "rear_projection",
    "lateral_distance",
    "selection_score",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_box_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int]:
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def extract_detector_candidates(result) -> tuple[list[dict], list[dict]]:
    if result.boxes is None or len(result.boxes) == 0:
        return [], []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = None

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    flock_candidates: list[dict] = []
    dog_candidates: list[dict] = []

    for index, box in enumerate(boxes_xyxy):
        class_id = int(class_ids[index])

        if class_id not in {0, 1}:
            continue

        x1, y1, x2, y2 = box.tolist()
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        candidate = {
            "box": (x1, y1, x2, y2),
            "center": calculate_box_center(x1, y1, x2, y2),
            "confidence": float(confidences[index]),
            "track_id": None if track_ids is None else int(track_ids[index]),
            "width": width,
            "height": height,
            "area": width * height,
        }

        if class_id == 0:
            flock_candidates.append(candidate)
        else:
            dog_candidates.append(candidate)

    return flock_candidates, dog_candidates


def select_rear_dog_closest_to_flock(
    dog_candidates: list[dict],
    flock_center: tuple[int, int],
    rear_direction: tuple[float, float] | np.ndarray | None,
    rear_direction_confidence: float,
    rear_direction_source: str,
    rear_projection_threshold: float,
    lateral_penalty: float,
) -> dict | None:
    if not dog_candidates or rear_direction is None:
        return None

    rear_direction = np.array(
        rear_direction,
        dtype=float,
    )

    behind_candidates = []

    for candidate in dog_candidates:
        dog_x, dog_y = candidate["center"]
        vector = np.array(
            [dog_x - flock_center[0], dog_y - flock_center[1]],
            dtype=float,
        )
        rear_projection = float(np.dot(vector, rear_direction))

        if rear_projection < rear_projection_threshold:
            continue

        lateral_vector = vector - rear_projection * rear_direction
        lateral_distance = float(np.linalg.norm(lateral_vector))
        distance_to_flock = float(np.linalg.norm(vector))

        enriched_candidate = candidate.copy()
        enriched_candidate["rear_projection"] = rear_projection
        enriched_candidate["lateral_distance"] = lateral_distance
        enriched_candidate["rear_direction_x"] = float(rear_direction[0])
        enriched_candidate["rear_direction_y"] = float(rear_direction[1])
        enriched_candidate["rear_direction_confidence"] = rear_direction_confidence
        enriched_candidate["rear_direction_source"] = rear_direction_source
        enriched_candidate["selection_score"] = (
            rear_projection - lateral_penalty * lateral_distance
        )
        enriched_candidate["distance_to_flock"] = distance_to_flock
        behind_candidates.append(enriched_candidate)

    if not behind_candidates:
        return None

    return min(
        behind_candidates,
        key=lambda candidate: candidate["distance_to_flock"],
    )


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
) -> dict:
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
        "flock_track_id": "" if flock_candidate["track_id"] is None else flock_candidate["track_id"],
        "dog_track_id": "" if dog_candidate["track_id"] is None else dog_candidate["track_id"],
        "flock_center_x": flock_center_x,
        "flock_center_y": flock_center_y,
        "dog_center_x": dog_center_x,
        "dog_center_y": dog_center_y,
        "flock_box_x1": flock_x1,
        "flock_box_y1": flock_y1,
        "flock_box_x2": flock_x2,
        "flock_box_y2": flock_y2,
        "flock_ellipse_center_x": flock_candidate["ellipse"][0],
        "flock_ellipse_center_y": flock_candidate["ellipse"][1],
        "flock_ellipse_major_axis": flock_candidate["ellipse"][2],
        "flock_ellipse_minor_axis": flock_candidate["ellipse"][3],
        "flock_ellipse_angle": flock_candidate["ellipse"][4],
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
        "rear_direction_x": dog_candidate["rear_direction_x"],
        "rear_direction_y": dog_candidate["rear_direction_y"],
        "rear_direction_confidence": dog_candidate["rear_direction_confidence"],
        "rear_direction_source": dog_candidate["rear_direction_source"],
        "rear_projection": dog_candidate["rear_projection"],
        "lateral_distance": dog_candidate["lateral_distance"],
        "selection_score": dog_candidate["selection_score"],
    }


def build_flock_only_tracking_dataset(
    video_paths: list[Path],
    model_path: Path,
    confidence: float,
    image_size: int,
    motion_window: int,
    motion_dead_zone: float,
    rear_projection_threshold: float,
    lateral_penalty: float,
    output_csv: Path,
) -> Path:
    resolved_videos = [resolve_project_path(video_path) for video_path in video_paths]

    for video_path in resolved_videos:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

    model_path = resolve_project_path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    model = load_detector(model_path)
    total_frames = 0
    total_rows = 0

    print("=" * 70)
    print("BUILDING FLOCK-ONLY TARGET DATASET")
    print("=" * 70)
    print(f"Videos: {len(resolved_videos)}")
    print(f"Detector: {model_path}")
    print(f"Output CSV: {output_csv}")
    print()

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DATASET_FIELDNAMES)
        writer.writeheader()

        for video_index, video_path in enumerate(resolved_videos):
            capture = cv2.VideoCapture(str(video_path))

            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                capture.release()
                raise RuntimeError(f"Invalid FPS in {video_path}")

            video_id = f"video_{video_index:02d}_{video_path.stem}"
            active_flock_track_id = None
            last_rear_direction = None
            rear_direction_confidence = 0.0
            rear_direction_source = "UNKNOWN"
            previous_ellipse_angle = None
            flock_trajectory: deque[tuple[int, int]] = deque(
                maxlen=max(2 * motion_window, 60),
            )
            current_segment_id = 0
            previous_visible_track_id = None
            frame_index = 0

            print(f"Processing {video_id}: {video_path.name}")

            while True:
                success, frame = capture.read()

                if not success:
                    break

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    classes=[0, 1],
                    conf=confidence,
                    iou=0.5,
                    imgsz=image_size,
                    verbose=False,
                )[0]

                flock_candidates, dog_candidates = extract_detector_candidates(result)
                selected_flock = select_main_flock(
                    candidates=flock_candidates,
                    active_track_id=active_flock_track_id,
                    single_flock=True,
                )

                if selected_flock is not None:
                    active_flock_track_id = selected_flock["track_id"]
                    flock_trajectory.append(selected_flock["center"])
                    (
                        last_rear_direction,
                        rear_direction_confidence,
                        rear_direction_source,
                    ) = calculate_stable_rear_direction(
                        trajectory=flock_trajectory,
                        window=motion_window,
                        dead_zone=motion_dead_zone,
                        last_rear_direction=last_rear_direction,
                    )

                    selected_dog = select_rear_dog_closest_to_flock(
                        dog_candidates=dog_candidates,
                        flock_center=selected_flock["center"],
                        rear_direction=last_rear_direction,
                        rear_direction_confidence=rear_direction_confidence,
                        rear_direction_source=rear_direction_source,
                        rear_projection_threshold=rear_projection_threshold,
                        lateral_penalty=lateral_penalty,
                    )

                    if selected_dog is not None:
                        current_track_id = selected_dog["track_id"]

                        front_direction = (
                            -last_rear_direction[0],
                            -last_rear_direction[1],
                        )
                        selected_flock = selected_flock.copy()
                        selected_flock["ellipse"] = calculate_flock_ellipse(
                            x1=selected_flock["box"][0],
                            y1=selected_flock["box"][1],
                            x2=selected_flock["box"][2],
                            y2=selected_flock["box"][3],
                            dx=float(front_direction[0]),
                            dy=float(front_direction[1]),
                            previous_angle=previous_ellipse_angle,
                            angle_dead_zone=0.0,
                            angle_smoothing=0.12,
                        )
                        previous_ellipse_angle = selected_flock["ellipse"][4]

                        if current_track_id != previous_visible_track_id:
                            current_segment_id += 1

                        writer.writerow(
                            build_frame_record(
                                video_id=video_id,
                                video_path=video_path,
                                frame_index=frame_index,
                                fps=fps,
                                frame_width=frame_width,
                                frame_height=frame_height,
                                flock_candidate=selected_flock,
                                dog_candidate=selected_dog,
                                segment_id=current_segment_id,
                            )
                        )
                        total_rows += 1
                        previous_visible_track_id = current_track_id
                    else:
                        previous_visible_track_id = None
                else:
                    previous_visible_track_id = None

                frame_index += 1
                total_frames += 1

                if frame_index % 150 == 0:
                    print(f"  {frame_index}/{frame_count} frames")

            capture.release()

    print()
    print("=" * 70)
    print("FLOCK-ONLY TARGET DATASET READY")
    print("=" * 70)
    print(f"Frames processed: {total_frames}")
    print(f"Tracked rows: {total_rows}")
    print(f"CSV: {output_csv}")

    return output_csv


def filter_rows_to_single_behind_dog(rows: list[dict]) -> tuple[list[dict], int]:
    frame_groups: dict[tuple[str, str, int], list[dict]] = {}

    for row in rows:
        key = (
            row.get("video_id", ""),
            row.get("segment_id", ""),
            int(row["frame"]),
        )
        frame_groups.setdefault(key, []).append(row)

    filtered_rows = []
    removed_rows = 0

    for key in sorted(frame_groups, key=lambda item: (item[0], item[1], item[2])):
        candidates = frame_groups[key]

        if len(candidates) == 1:
            filtered_rows.append(candidates[0])
            continue

        selected_row = min(
            candidates,
            key=lambda row: math.hypot(
                float(row["dog_center_x"]) - float(row["flock_center_x"]),
                float(row["dog_center_y"]) - float(row["flock_center_y"]),
            ),
        )

        filtered_rows.append(selected_row)
        removed_rows += len(candidates) - 1

    return filtered_rows, removed_rows


def write_filtered_dataset_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
    seed: int,
) -> tuple[FlockTargetGRUPredictor, float]:
    dataset = TemporalWindowDataset(train_features, train_targets)

    if len(dataset) < 2:
        raise RuntimeError("Not enough windows to train the predictor.")

    validation_size = max(1, int(round(0.1 * len(dataset))))
    training_size = max(1, len(dataset) - validation_size)
    validation_size = len(dataset) - training_size

    generator = torch.Generator().manual_seed(seed)
    training_subset, validation_subset = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)

    model = FlockTargetGRUPredictor(train_features.shape[-1], hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features_batch), targets_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        validation_losses = []

        with torch.no_grad():
            for features_batch, targets_batch in validation_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                validation_losses.append(float(criterion(model(features_batch), targets_batch).item()))

        mean_validation_loss = float(np.mean(validation_losses))

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | val={mean_validation_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    return model, best_validation_loss


def evaluate_model(
    model: FlockTargetGRUPredictor,
    features: np.ndarray,
    metadata: list,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    if len(metadata) == 0:
        return {"mae_px": math.nan, "mae_norm": math.nan}

    normalized_features, _, _ = normalize_features(features, mean=mean, std=std)

    with torch.no_grad():
        predictions = model(torch.from_numpy(normalized_features).to(device)).cpu().numpy()

    pixel_errors = []
    normalized_errors = []

    for prediction, sample_metadata in zip(predictions, metadata):
        predicted_x, predicted_y = denormalize_target(sample_metadata.current_row, prediction)
        real_x = float(sample_metadata.future_row["dog_center_x"])
        real_y = float(sample_metadata.future_row["dog_center_y"])
        target = build_target_vector(
            sample_metadata.current_row,
            sample_metadata.future_row,
        )

        pixel_errors.append(math.hypot(predicted_x - real_x, predicted_y - real_y))
        normalized_errors.append(
            math.hypot(
                float(prediction[0]) - float(target[0]),
                float(prediction[1]) - float(target[1]),
            )
        )

    return {"mae_px": float(np.mean(pixel_errors)), "mae_norm": float(np.mean(normalized_errors))}


def train_flock_target_predictor(
    dataset_csv: Path | None,
    video_paths: list[Path] | None,
    detector_model_path: Path,
    detector_confidence: float,
    detector_image_size: int,
    motion_window: int,
    motion_dead_zone: float,
    rear_projection_threshold: float,
    lateral_penalty: float,
    history_length: int,
    prediction_horizon_seconds: float,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device_name: str,
    seed: int,
    single_behind_dog: bool,
    run_name: str,
) -> None:
    set_seed(seed)

    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    if dataset_csv is None:
        if not video_paths:
            raise ValueError("Debes pasar --dataset o --videos.")

        dataset_csv = build_flock_only_tracking_dataset(
            video_paths=video_paths,
            model_path=detector_model_path,
            confidence=detector_confidence,
            image_size=detector_image_size,
            motion_window=motion_window,
            motion_dead_zone=motion_dead_zone,
            rear_projection_threshold=rear_projection_threshold,
            lateral_penalty=lateral_penalty,
            output_csv=output_directory / "flock_only_tracking_dataset.csv",
        )

    dataset_csv = resolve_project_path(dataset_csv)

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    rows = load_rows(dataset_csv)

    if not rows:
        raise RuntimeError("The dataset CSV is empty.")

    if single_behind_dog:
        original_row_count = len(rows)
        rows, removed_rows = filter_rows_to_single_behind_dog(rows)
        filtered_csv_path = output_directory / "single_behind_dog_dataset.csv"
        write_filtered_dataset_csv(rows, filtered_csv_path)

        print(
            "Single behind dog filter: "
            f"{original_row_count} -> {len(rows)} rows "
            f"({removed_rows} removed)"
        )
        print(f"Filtered dataset CSV: {filtered_csv_path}")

    rows_by_video = group_rows_by_video(rows)
    video_ids = sorted(rows_by_video.keys())

    if len(video_ids) < 2:
        raise RuntimeError("At least two videos are required for leave-one-video-out evaluation.")

    fps_values = {}

    for video_id, video_rows in rows_by_video.items():
        if len(video_rows) < 2:
            continue
        delta_time = float(video_rows[1]["time_seconds"]) - float(video_rows[0]["time_seconds"])
        fps_values[video_id] = 1.0 / max(1e-6, delta_time)

    if not fps_values:
        raise RuntimeError("Could not infer FPS from the temporal dataset.")

    prediction_offset_frames = max(1, int(round(prediction_horizon_seconds * float(np.median(list(fps_values.values()))))))

    metrics_path = output_directory / "leave_one_video_out_metrics.json"
    checkpoint_path = output_directory / "flock_target_predictor.pt"
    model_copy = MODELS_ROOT / f"{run_name}_flock_target_predictor.pt"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    print("=" * 70)
    print("TRAINING FLOCK-ONLY TARGET PREDICTOR")
    print("=" * 70)
    print(f"Dataset: {dataset_csv}")
    print(f"Single behind dog: {single_behind_dog}")
    print(f"Prediction offset: {prediction_offset_frames} frames")
    print()

    fold_results = []

    for held_out_video in video_ids:
        train_videos = set(video_ids) - {held_out_video}
        train_features, train_targets, _ = build_windows(rows_by_video, history_length, prediction_offset_frames, train_videos)
        test_features, _, test_metadata = build_windows(rows_by_video, history_length, prediction_offset_frames, {held_out_video})

        if len(train_features) == 0 or len(test_features) == 0:
            continue

        normalized_train_features, mean, std = normalize_features(train_features)
        model, best_validation_loss = train_one_model(
            normalized_train_features,
            train_targets,
            batch_size,
            learning_rate,
            epochs,
            hidden_size,
            num_layers,
            dropout,
            device,
            seed,
        )

        metrics = evaluate_model(model, test_features, test_metadata, mean, std, device)
        fold_results.append(
            {
                "held_out_video": held_out_video,
                "train_windows": int(len(train_features)),
                "test_windows": int(len(test_features)),
                "best_validation_loss": float(best_validation_loss),
                "mae_px": metrics["mae_px"],
                "mae_norm": metrics["mae_norm"],
            }
        )

    all_features, all_targets, _ = build_windows(rows_by_video, history_length, prediction_offset_frames, set(video_ids))

    if len(all_features) == 0:
        raise RuntimeError("No training windows could be built from the dataset.")

    normalized_all_features, final_mean, final_std = normalize_features(all_features)
    final_model, final_validation_loss = train_one_model(
        normalized_all_features,
        all_targets,
        batch_size,
        learning_rate,
        epochs,
        hidden_size,
        num_layers,
        dropout,
        device,
        seed,
    )

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "target_representation": "rear_lateral_ellipse_v1",
        "feature_mean": final_mean,
        "feature_std": final_std,
        "history_length": history_length,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "prediction_offset_frames": prediction_offset_frames,
        "input_size": int(all_features.shape[-1]),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "video_ids": video_ids,
        "final_validation_loss": float(final_validation_loss),
        "leave_one_video_out": fold_results,
    }

    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, model_copy)

    summary = {
        "dataset_csv": str(dataset_csv),
        "videos": video_ids,
        "history_length": history_length,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "prediction_offset_frames": prediction_offset_frames,
        "target_representation": "rear_lateral_ellipse_v1",
        "single_behind_dog": single_behind_dog,
        "all_windows": int(len(all_features)),
        "final_validation_loss": float(final_validation_loss),
        "average_mae_px": float(np.nanmean([result["mae_px"] for result in fold_results])) if fold_results else math.nan,
        "average_mae_norm": float(np.nanmean([result["mae_norm"] for result in fold_results])) if fold_results else math.nan,
        "leave_one_video_out": fold_results,
        "checkpoint": str(checkpoint_path),
        "model_copy": str(model_copy),
    }

    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model copy: {model_copy}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a flock-only GRU target predictor.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--videos", type=Path, nargs="+", default=None)
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL_PATH)
    parser.add_argument("--detector-confidence", type=float, default=0.15)
    parser.add_argument("--detector-image-size", type=int, default=960)
    parser.add_argument("--motion-window", type=int, default=12)
    parser.add_argument("--motion-dead-zone", type=float, default=10.0)
    parser.add_argument("--rear-projection-threshold", type=float, default=15.0)
    parser.add_argument("--lateral-penalty", type=float, default=0.35)
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--prediction-horizon-seconds", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-behind-dog",
        action="store_true",
        help=(
            "Deja un solo perro por frame: el que esta detras del rebaño "
            "y, si hay varios, el mas cercano."
        ),
    )
    parser.add_argument("--name", type=str, default="flock_target_gru")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    train_flock_target_predictor(
        dataset_csv=args.dataset,
        video_paths=args.videos,
        detector_model_path=args.detector_model,
        detector_confidence=args.detector_confidence,
        detector_image_size=args.detector_image_size,
        motion_window=args.motion_window,
        motion_dead_zone=args.motion_dead_zone,
        rear_projection_threshold=args.rear_projection_threshold,
        lateral_penalty=args.lateral_penalty,
        history_length=args.history_length,
        prediction_horizon_seconds=args.prediction_horizon_seconds,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device_name=args.device,
        seed=args.seed,
        single_behind_dog=args.single_behind_dog,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
