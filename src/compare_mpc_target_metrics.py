from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


DISTANCE_COLUMN = "mpc_dog_distance_flock_units"
MPC_FLOCK_DISTANCE_COLUMN = "mpc_flock_distance_flock_units"
DOG_FLOCK_DISTANCE_COLUMN = "dog_flock_distance_flock_units"


def load_valid_distances(path: Path) -> tuple[dict[int, float], dict[int, float], dict[int, float], int]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return {}, {}, {}, 0
    required = {"frame", "metric_valid", DISTANCE_COLUMN, MPC_FLOCK_DISTANCE_COLUMN, DOG_FLOCK_DISTANCE_COLUMN}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} does not contain metric columns: {', '.join(sorted(missing))}")

    dog_distances = {}
    flock_distances = {}
    dog_flock_distances = {}
    for row in rows:
        if row["metric_valid"] != "1" or not row[DISTANCE_COLUMN] or not row[MPC_FLOCK_DISTANCE_COLUMN] or not row[DOG_FLOCK_DISTANCE_COLUMN]:
            continue
        frame = int(row["frame"])
        dog_distances[frame] = float(row[DISTANCE_COLUMN])
        flock_distances[frame] = float(row[MPC_FLOCK_DISTANCE_COLUMN])
        dog_flock_distances[frame] = float(row[DOG_FLOCK_DISTANCE_COLUMN])
    return dog_distances, flock_distances, dog_flock_distances, len(rows)


def describe(values: list[float]) -> dict[str, str]:
    if not values:
        return {"median": "", "mean": ""}
    ordered = sorted(values)
    return {
        "median": f"{statistics.median(ordered):.6f}",
        "mean": f"{statistics.mean(ordered):.6f}",
    }


def build_row(
    video_id: str,
    method: str,
    total_rows: int,
    all_valid: dict[int, float],
    paired_values: list[float],
    paired_flock_values: list[float],
    paired_dog_flock_values: list[float],
    paired_frames: int,
    winner: str,
    flock_win_rate: str,
) -> dict[str, str | int]:
    return {
        "video_id": video_id,
        "method": method,
        "csv_frames": total_rows,
        "valid_metric_frames": len(all_valid),
        "valid_metric_rate": f"{len(all_valid) / total_rows:.6f}" if total_rows else "",
        "paired_frames": paired_frames,
        "median_mpc_dog_distance_flock_units": describe(paired_values)["median"],
        "mean_mpc_dog_distance_flock_units": describe(paired_values)["mean"],
        "median_mpc_flock_distance_flock_units": describe(paired_flock_values)["median"],
        "median_dog_flock_distance_flock_units": describe(paired_dog_flock_values)["median"],
        "winner_by_paired_median": winner,
        "flock_only_frame_win_rate": flock_win_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired flock-normalized MPC-target-to-dog distances from two video CSVs."
    )
    parser.add_argument("--flock-csv", type=Path, required=True, action="append", help="Flock-only CSV. Repeat once per video.")
    parser.add_argument("--mpc-csv", type=Path, required=True, action="append", help="MPC-herding CSV in the same order. Repeat once per video.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--video-id", type=str, action="append", default=None, help="Optional ID in the same order as each CSV pair.")
    args = parser.parse_args()

    if len(args.flock_csv) != len(args.mpc_csv):
        parser.error("Provide the same number of --flock-csv and --mpc-csv arguments.")
    if args.video_id is not None and len(args.video_id) != len(args.flock_csv):
        parser.error("Provide either no --video-id values or one for every CSV pair.")

    rows = []
    comparable_frames = 0
    for index, (flock_csv, mpc_csv) in enumerate(zip(args.flock_csv, args.mpc_csv)):
        flock_distances, flock_center_distances, flock_dog_flock_distances, flock_rows = load_valid_distances(flock_csv)
        mpc_distances, mpc_center_distances, mpc_dog_flock_distances, mpc_rows = load_valid_distances(mpc_csv)
        paired_frames = sorted(set(flock_distances) & set(mpc_distances))
        flock_paired = [flock_distances[frame] for frame in paired_frames]
        mpc_paired = [mpc_distances[frame] for frame in paired_frames]
        flock_center_paired = [flock_center_distances[frame] for frame in paired_frames]
        mpc_center_paired = [mpc_center_distances[frame] for frame in paired_frames]
        flock_dog_flock_paired = [flock_dog_flock_distances[frame] for frame in paired_frames]
        mpc_dog_flock_paired = [mpc_dog_flock_distances[frame] for frame in paired_frames]
        video_id = args.video_id[index] if args.video_id is not None else flock_csv.stem
        comparable_frames += len(paired_frames)

        if not paired_frames:
            winner = "NO_COMPARABLE_FRAMES"
            flock_win_rate = ""
        else:
            flock_median = statistics.median(flock_paired)
            mpc_median = statistics.median(mpc_paired)
            winner = "TIE" if flock_median == mpc_median else ("flock_only_target_prediction" if flock_median < mpc_median else "mpc_herding")
            flock_win_rate = f"{sum(flock < mpc for flock, mpc in zip(flock_paired, mpc_paired)) / len(paired_frames):.6f}"

        rows.extend(
            [
                build_row(video_id, "flock_only_target_prediction", flock_rows, flock_distances, flock_paired, flock_center_paired, flock_dog_flock_paired, len(paired_frames), winner, flock_win_rate),
                build_row(video_id, "mpc_herding", mpc_rows, mpc_distances, mpc_paired, mpc_center_paired, mpc_dog_flock_paired, len(paired_frames), winner, flock_win_rate),
            ]
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparable frames: {comparable_frames}")
    print(f"Summary: {args.output_csv}")


if __name__ == "__main__":
    main()
