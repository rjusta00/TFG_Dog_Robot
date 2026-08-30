import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split


SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rear_dog_prediction.common import (
    OUTPUT_ROOT,
    denormalize_prediction,
    group_rows_by_video,
    load_tracking_rows,
    resolve_project_path,
)
from rear_dog_prediction.model import (
    RearDogGRUPredictor,
    TemporalWindowDataset,
    build_windows,
    normalize_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_data_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TemporalWindowDataset(
        features=features,
        targets=targets,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


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
) -> tuple[RearDogGRUPredictor, float]:
    dataset = TemporalWindowDataset(
        features=train_features,
        targets=train_targets,
    )

    if len(dataset) < 2:
        raise RuntimeError(
            "Not enough windows to train the predictor."
        )

    validation_size = max(
        1,
        int(round(0.1 * len(dataset))),
    )

    training_size = len(dataset) - validation_size

    if training_size < 1:
        training_size = len(dataset) - 1
        validation_size = 1

    generator = torch.Generator().manual_seed(seed)
    training_subset, validation_subset = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        training_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = RearDogGRUPredictor(
        input_size=train_features.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.SmoothL1Loss()
    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            predictions = model(features_batch)
            loss = criterion(predictions, targets_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []

        with torch.no_grad():
            for features_batch, targets_batch in validation_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)
                predictions = model(features_batch)
                loss = criterion(predictions, targets_batch)
                validation_losses.append(float(loss.item()))

        mean_validation_loss = float(np.mean(validation_losses))

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            mean_train_loss = float(np.mean(train_losses))
            print(
                f"Epoch {epoch + 1}/{epochs} | train={mean_train_loss:.6f} | val={mean_validation_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training finished without a valid checkpoint."
        )

    model.load_state_dict(best_state)
    return model, best_validation_loss


def evaluate_model(
    model: RearDogGRUPredictor,
    features: np.ndarray,
    metadata: list,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    if len(metadata) == 0:
        return {
            "mae_px": math.nan,
            "mae_norm": math.nan,
        }

    normalized_features, _, _ = normalize_features(
        features,
        mean=mean,
        std=std,
    )

    model.eval()

    with torch.no_grad():
        predictions = model(
            torch.from_numpy(normalized_features).to(device)
        ).cpu().numpy()

    pixel_errors = []
    normalized_errors = []

    for prediction, sample_metadata in zip(predictions, metadata):
        predicted_x, predicted_y = denormalize_prediction(
            current_row=sample_metadata.current_row,
            prediction=prediction,
        )

        real_x = float(sample_metadata.future_row["dog_center_x"])
        real_y = float(sample_metadata.future_row["dog_center_y"])

        pixel_errors.append(
            math.hypot(
                predicted_x - real_x,
                predicted_y - real_y,
            )
        )

        normalized_errors.append(
            math.hypot(
                float(prediction[0])
                - (
                    (real_x - float(sample_metadata.current_row["flock_center_x"]))
                    / max(1.0, float(sample_metadata.current_row["flock_width"]))
                ),
                float(prediction[1])
                - (
                    (real_y - float(sample_metadata.current_row["flock_center_y"]))
                    / max(1.0, float(sample_metadata.current_row["flock_height"]))
                ),
            )
        )

    return {
        "mae_px": float(np.mean(pixel_errors)),
        "mae_norm": float(np.mean(normalized_errors)),
    }


def train_rear_dog_predictor(
    dataset_csv: Path,
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
    run_name: str,
) -> None:
    set_seed(seed)

    dataset_csv = resolve_project_path(
        dataset_csv
    )

    if not dataset_csv.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {dataset_csv}"
        )

    rows = load_tracking_rows(
        dataset_csv
    )

    if not rows:
        raise RuntimeError(
            "The dataset CSV is empty."
        )

    rows_by_video = group_rows_by_video(
        rows
    )
    video_ids = sorted(rows_by_video.keys())

    if len(video_ids) < 2:
        raise RuntimeError(
            "At least two videos are required for leave-one-video-out evaluation."
        )

    fps_values: dict[str, float] = {}

    for video_id, video_rows in rows_by_video.items():
        if len(video_rows) < 2:
            continue

        delta_time = (
            float(video_rows[1]["time_seconds"])
            - float(video_rows[0]["time_seconds"])
        )

        fps_values[video_id] = 1.0 / max(
            1e-6,
            delta_time,
        )

    if not fps_values:
        raise RuntimeError(
            "Could not infer FPS from the temporal dataset."
        )

    median_fps = float(np.median(list(fps_values.values())))
    prediction_offset_frames = max(
        1,
        int(round(prediction_horizon_seconds * median_fps)),
    )

    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = output_directory / "leave_one_video_out_metrics.json"
    checkpoint_path = output_directory / "rear_dog_predictor.pt"
    final_model_copy = MODELS_ROOT / f"{run_name}_rear_dog_predictor.pt"
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)

    print("=" * 70)
    print("TRAINING REAR DOG PREDICTOR")
    print("=" * 70)
    print(f"Dataset: {dataset_csv}")
    print(f"Videos: {len(video_ids)}")
    print(f"History length: {history_length}")
    print(f"Prediction horizon: {prediction_horizon_seconds:.2f} s")
    print(f"Prediction offset: {prediction_offset_frames} frames")
    print(f"Device: {device}")
    print()

    leave_one_video_out_results = []

    for held_out_video_id in video_ids:
        train_video_ids = set(video_ids) - {held_out_video_id}

        train_features, train_targets, _ = build_windows(
            rows_by_video=rows_by_video,
            history_length=history_length,
            prediction_offset_frames=prediction_offset_frames,
            allowed_videos=train_video_ids,
        )

        test_features, _, test_metadata = build_windows(
            rows_by_video=rows_by_video,
            history_length=history_length,
            prediction_offset_frames=prediction_offset_frames,
            allowed_videos={held_out_video_id},
        )

        if len(train_features) == 0 or len(test_features) == 0:
            print(
                f"Skipping fold for {held_out_video_id}: not enough windows."
            )
            continue

        normalized_train_features, mean, std = normalize_features(
            train_features
        )

        model, best_validation_loss = train_one_model(
            train_features=normalized_train_features,
            train_targets=train_targets,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
            seed=seed,
        )

        metrics = evaluate_model(
            model=model,
            features=test_features,
            metadata=test_metadata,
            mean=mean,
            std=std,
            device=device,
        )

        fold_result = {
            "held_out_video": held_out_video_id,
            "train_windows": int(len(train_features)),
            "test_windows": int(len(test_features)),
            "best_validation_loss": float(best_validation_loss),
            "mae_px": metrics["mae_px"],
            "mae_norm": metrics["mae_norm"],
        }

        leave_one_video_out_results.append(fold_result)
        print(json.dumps(fold_result, indent=2))

    all_features, all_targets, _ = build_windows(
        rows_by_video=rows_by_video,
        history_length=history_length,
        prediction_offset_frames=prediction_offset_frames,
        allowed_videos=set(video_ids),
    )

    if len(all_features) == 0:
        raise RuntimeError(
            "No training windows could be built from the dataset."
        )

    normalized_all_features, final_mean, final_std = normalize_features(
        all_features
    )

    final_model, final_validation_loss = train_one_model(
        train_features=normalized_all_features,
        train_targets=all_targets,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        seed=seed,
    )

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
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
        "leave_one_video_out": leave_one_video_out_results,
    }

    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, final_model_copy)

    summary = {
        "dataset_csv": str(dataset_csv),
        "videos": video_ids,
        "history_length": history_length,
        "prediction_horizon_seconds": prediction_horizon_seconds,
        "prediction_offset_frames": prediction_offset_frames,
        "all_windows": int(len(all_features)),
        "final_validation_loss": float(final_validation_loss),
        "average_mae_px": float(np.nanmean([result["mae_px"] for result in leave_one_video_out_results])) if leave_one_video_out_results else math.nan,
        "average_mae_norm": float(np.nanmean([result["mae_norm"] for result in leave_one_video_out_results])) if leave_one_video_out_results else math.nan,
        "leave_one_video_out": leave_one_video_out_results,
        "checkpoint": str(checkpoint_path),
        "model_copy": str(final_model_copy),
    }

    metrics_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("REAR DOG PREDICTOR TRAINED")
    print("=" * 70)
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model copy: {final_model_copy}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GRU predictor for rear dog motion using multiple videos."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--history-length",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--prediction-horizon-seconds",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="rear_dog_gru",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_rear_dog_predictor(
        dataset_csv=args.dataset,
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
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
