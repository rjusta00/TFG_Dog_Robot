from __future__ import annotations

import argparse
import csv
import html
import statistics
from pathlib import Path

try:
    from .gpu_telemetry import GPU_TELEMETRY_COLUMNS
except ImportError:  # Supports execution as `python src/compare_mpc_target_metrics.py`.
    from gpu_telemetry import GPU_TELEMETRY_COLUMNS


DISTANCE_COLUMN = "mpc_dog_distance_px"
MPC_FLOCK_DISTANCE_COLUMN = "mpc_flock_distance_px"
DOG_FLOCK_DISTANCE_COLUMN = "dog_flock_distance_px"
PERFORMANCE_COLUMNS = (
    "inference_seconds",
    "control_seconds",
    "frame_seconds",
    "python_memory_peak_mb",
    "control_energy_proxy",
    *GPU_TELEMETRY_COLUMNS,
)
METRIC_SUMMARY_COLUMNS = (
    "video_id",
    "method",
    "csv_frames",
    "valid_metric_frames",
    "valid_metric_rate",
    "paired_frames",
    "median_mpc_dog_distance_px",
    "mean_mpc_dog_distance_px",
    "median_mpc_flock_distance_px",
    "median_dog_flock_distance_px",
    "winner_by_paired_median",
    "flock_only_frame_win_rate",
    "flock_detected_frames",
    "flock_detected_rate",
    "dog_detected_frames",
    "dog_detected_rate",
)


def performance_csv_path(spatial_csv_path: Path) -> Path:
    return spatial_csv_path.with_name(f"{spatial_csv_path.stem}_performance.csv")


def comparison_performance_csv_path(metrics_csv_path: Path) -> Path:
    return metrics_csv_path.with_name(f"{metrics_csv_path.stem}_performance.csv")


def write_rows_with_video_spacers(
    csv_file,
    writer: csv.DictWriter,
    rows: list[dict[str, str | int]],
    columns: tuple[str, ...] | list[str],
) -> None:
    previous_video_id = None
    for row in rows:
        if previous_video_id is not None and row["video_id"] != previous_video_id:
            csv_file.write("\n")
        writer.writerow({column: row[column] for column in columns})
        previous_video_id = row["video_id"]


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


def load_paired_performance(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return {}
    required = {"frame", *PERFORMANCE_COLUMNS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} does not contain performance columns: {', '.join(sorted(missing))}")

    performance = {}
    for row in rows:
        if any(not row[column] for column in PERFORMANCE_COLUMNS):
            continue
        performance[int(row["frame"])] = {column: float(row[column]) for column in PERFORMANCE_COLUMNS}
    return performance


def load_detection_summary(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        return 0, 0
    required = {"flock_detected", "dog_detections"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} does not contain detection columns: {', '.join(sorted(missing))}")
    return (
        sum(row["flock_detected"] == "1" for row in rows),
        sum(int(row["dog_detections"]) > 0 for row in rows),
    )


def write_grouped_bar_chart(
    path: Path,
    title: str,
    rows: list[dict[str, str | int]],
    columns: list[tuple[str, str]],
    y_label: str,
) -> None:
    width, height = 1280, 720
    left, right, top, bottom = 100, 40, 80, 150
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [[float(row.get(column, 0) or 0) for column, _ in columns] for row in rows]
    maximum = max((value for row_values in values for value in row_values), default=0.0)
    maximum = maximum * 1.1 if maximum else 1.0
    colors = ["#177E89", "#DB3A34", "#FFC857"]
    group_width = plot_width / max(1, len(rows))
    bar_width = group_width / (len(columns) + 1)
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
        f'<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="16">{html.escape(y_label)}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_height - plot_height * tick / 5
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ddd"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{value:.3g}</text>')
    for index, (row, row_values) in enumerate(zip(rows, values)):
        group_start = left + index * group_width
        for metric_index, value in enumerate(row_values):
            bar_height = value / maximum * plot_height
            x = group_start + (metric_index + 0.5) * bar_width
            y = top + plot_height - bar_height
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.8:.1f}" height="{bar_height:.1f}" fill="{colors[metric_index % len(colors)]}"/>')
        label = f'{row["video_id"]}: {row["method"]}'
        x = group_start + group_width / 2
        elements.append(f'<text x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(30 {x:.1f} {top + plot_height + 22})">{html.escape(str(label))}</text>')
    for index, (_, label) in enumerate(columns):
        x = left + index * 220
        elements.append(f'<rect x="{x}" y="{height - 42}" width="16" height="16" fill="{colors[index % len(colors)]}"/>')
        elements.append(f'<text x="{x + 22}" y="{height - 29}" font-family="Arial" font-size="14">{html.escape(label)}</text>')
    path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\">" + "".join(elements) + "</svg>", encoding="utf-8")


def write_result_charts(output_csv: Path, rows: list[dict[str, str | int]]) -> list[Path]:
    chart_specs = (
        ("target_accuracy", "Target-to-Dog Distance", [("mean_mpc_dog_distance_px", "Mean distance"), ("median_mpc_dog_distance_px", "Median distance")], "pixels"),
        ("timing", "Mean Processing Time", [("mean_inference_seconds", "Inference"), ("mean_control_seconds", "MPC control"), ("mean_frame_seconds", "Total frame")], "seconds"),
        ("detections", "Detection Coverage", [("flock_detected_rate", "Flock"), ("dog_detected_rate", "Dog")], "fraction of frames"),
        ("memory", "Peak Python Memory", [("mean_python_memory_peak_mb", "Mean peak memory")], "MB"),
        ("gpu_power", "Mean GPU Power", [("mean_gpu_power_w", "GPU power")], "W"),
        ("gpu_memory", "Maximum GPU Memory", [("max_gpu_memory_used_mb", "GPU memory")], "MB"),
        ("gpu_utilization", "Mean GPU Utilization", [("mean_gpu_utilization_pct", "GPU utilization")], "%"),
        ("gpu_temperature", "Maximum GPU Temperature", [("max_gpu_temperature_c", "GPU temperature")], "degrees C"),
        ("gpu_energy", "GPU Energy Per Run", [("run_gpu_energy_j", "GPU energy")], "J"),
        ("control_effort", "Total Control-Effort Proxy", [("total_control_energy_proxy", "Control effort")], "proxy units"),
    )
    paths = []
    for suffix, title, columns, y_label in chart_specs:
        chart_path = output_csv.with_name(f"{output_csv.stem}_{suffix}.svg")
        write_grouped_bar_chart(chart_path, title, rows, columns, y_label)
        paths.append(chart_path)
    return paths


def build_row(
    video_id: str,
    method: str,
    total_rows: int,
    all_valid: dict[int, float],
    paired_values: list[float],
    paired_flock_values: list[float],
    paired_dog_flock_values: list[float],
    paired_frames: int,
    performance: dict[str, list[float]],
    paired_performance_frames: int,
    flock_detected_frames: int,
    dog_detected_frames: int,
    winner: str,
    flock_win_rate: str,
) -> dict[str, str | int]:
    row: dict[str, str | int] = {
        "video_id": video_id,
        "method": method,
        "csv_frames": total_rows,
        "valid_metric_frames": len(all_valid),
        "valid_metric_rate": f"{len(all_valid) / total_rows:.6f}" if total_rows else "",
        "paired_frames": paired_frames,
        "median_mpc_dog_distance_px": describe(paired_values)["median"],
        "mean_mpc_dog_distance_px": describe(paired_values)["mean"],
        "median_mpc_flock_distance_px": describe(paired_flock_values)["median"],
        "median_dog_flock_distance_px": describe(paired_dog_flock_values)["median"],
        "winner_by_paired_median": winner,
        "flock_only_frame_win_rate": flock_win_rate,
    }
    row["paired_performance_frames"] = paired_performance_frames
    row["flock_detected_frames"] = flock_detected_frames
    row["flock_detected_rate"] = f"{flock_detected_frames / total_rows:.6f}" if total_rows else ""
    row["dog_detected_frames"] = dog_detected_frames
    row["dog_detected_rate"] = f"{dog_detected_frames / total_rows:.6f}" if total_rows else ""
    for column, values in performance.items():
        summary = describe(values)
        row[f"median_{column}"] = summary["median"]
        row[f"mean_{column}"] = summary["mean"]
        if column in {"inference_seconds", "control_seconds", "frame_seconds", "control_energy_proxy"}:
            row[f"total_{column}"] = f"{sum(values):.6f}" if values else ""
        if column in GPU_TELEMETRY_COLUMNS:
            row[f"max_{column}"] = f"{max(values):.6f}" if values else ""
        if column == "gpu_energy_j":
            row["run_gpu_energy_j"] = f"{max(values):.6f}" if values else ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired pixel distances from two video CSVs."
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
        flock_performance = load_paired_performance(performance_csv_path(flock_csv))
        mpc_performance = load_paired_performance(performance_csv_path(mpc_csv))
        flock_detections = load_detection_summary(flock_csv)
        mpc_detections = load_detection_summary(mpc_csv)
        paired_frames = sorted(set(flock_distances) & set(mpc_distances))
        flock_paired = [flock_distances[frame] for frame in paired_frames]
        mpc_paired = [mpc_distances[frame] for frame in paired_frames]
        flock_center_paired = [flock_center_distances[frame] for frame in paired_frames]
        mpc_center_paired = [mpc_center_distances[frame] for frame in paired_frames]
        flock_dog_flock_paired = [flock_dog_flock_distances[frame] for frame in paired_frames]
        mpc_dog_flock_paired = [mpc_dog_flock_distances[frame] for frame in paired_frames]
        video_id = args.video_id[index] if args.video_id is not None else flock_csv.stem
        comparable_frames += len(paired_frames)
        paired_performance_frames = sorted(set(flock_performance) & set(mpc_performance))
        flock_performance_values = {
            column: [flock_performance[frame][column] for frame in paired_performance_frames]
            for column in PERFORMANCE_COLUMNS
        }
        mpc_performance_values = {
            column: [mpc_performance[frame][column] for frame in paired_performance_frames]
            for column in PERFORMANCE_COLUMNS
        }

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
                build_row(video_id, "flock_only_target_prediction", flock_rows, flock_distances, flock_paired, flock_center_paired, flock_dog_flock_paired, len(paired_frames), flock_performance_values, len(paired_performance_frames), *flock_detections, winner, flock_win_rate),
                build_row(video_id, "mpc_herding", mpc_rows, mpc_distances, mpc_paired, mpc_center_paired, mpc_dog_flock_paired, len(paired_frames), mpc_performance_values, len(paired_performance_frames), *mpc_detections, winner, flock_win_rate),
            ]
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_SUMMARY_COLUMNS)
        writer.writeheader()
        write_rows_with_video_spacers(csv_file, writer, rows, METRIC_SUMMARY_COLUMNS)

    performance_output_csv = comparison_performance_csv_path(args.output_csv)
    performance_summary_columns = [
        "video_id",
        "method",
        "paired_performance_frames",
        *[column for column in rows[0] if column not in METRIC_SUMMARY_COLUMNS and column != "paired_performance_frames"],
    ]
    with performance_output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=performance_summary_columns)
        writer.writeheader()
        write_rows_with_video_spacers(csv_file, writer, rows, performance_summary_columns)

    chart_paths = write_result_charts(args.output_csv, rows)

    print(f"Comparable frames: {comparable_frames}")
    print(f"Summary: {args.output_csv}")
    print(f"Performance summary: {performance_output_csv}")
    for chart_path in chart_paths:
        print(f"Chart: {chart_path}")


if __name__ == "__main__":
    main()
