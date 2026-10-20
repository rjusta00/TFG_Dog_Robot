"""Run reproducible warm-up and performance trials for the two video pipelines."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_ROWS_TO_EXCLUDE = 30

RUN_FIELDS = (
    "phase",
    "repeat",
    "sequence",
    "model",
    "run_name",
    "started_at",
    "finished_at",
    "return_code",
    "wall_time_seconds",
    "command",
    "log_path",
    "result_csv_path",
    "performance_csv_path",
    "frames",
    "frame_seconds_total",
    "effective_fps",
    "frame_seconds_mean_steady",
    "frame_seconds_median_steady",
    "frame_seconds_p95_steady",
    "inference_seconds_total",
    "control_seconds_total",
    "gpu_energy_final_j",
    "gpu_memory_peak_mb",
    "gpu_temperature_peak_c",
    "optimizer_updates",
    "optimizer_success_rate",
    "summary_error",
)

SUMMARY_METRICS = (
    "wall_time_seconds",
    "effective_fps",
    "frame_seconds_total",
    "frame_seconds_median_steady",
    "frame_seconds_p95_steady",
    "gpu_energy_final_j",
    "gpu_memory_peak_mb",
    "gpu_temperature_peak_c",
    "optimizer_success_rate",
)

MODEL_LABELS = {
    "mpc_herding": "MPC herding",
    "flock_only_target_prediction": "Flock-only predictor",
}


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_coordinate(value: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        return float(x_text), float(y_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Coordinates must use the x,y format.") from error


def format_coordinate(value: tuple[float, float]) -> str:
    return f"{value[0]:g},{value[1]:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run warm-up and measurement trials for MPC herding and flock-only prediction."
    )
    parser.add_argument("--video", default="videos/rebano_02.mp4")
    parser.add_argument("--detector-model", default="models/dogRobot_v2_3_best.pt")
    parser.add_argument("--predictor-checkpoint", default="models/flock_target_gru_v3.pt")
    parser.add_argument("--robot-start", type=parse_coordinate, default=(80.0, 80.0), metavar="X,Y")
    parser.add_argument("--target", type=parse_coordinate, default=(1840.0, 650.0), metavar="X,Y")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--measurements", type=int, default=10)
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize_run(performance_path: Path, result_path: Path) -> dict[str, Any]:
    if not performance_path.is_file():
        raise FileNotFoundError(f"Performance CSV not found: {performance_path}")

    with performance_path.open("r", newline="", encoding="utf-8") as csv_file:
        performance_rows = list(csv.DictReader(csv_file))
    if not performance_rows:
        raise ValueError("Performance CSV is empty.")

    frame_seconds = [as_float(row.get("frame_seconds")) for row in performance_rows]
    inference_seconds = [as_float(row.get("inference_seconds")) for row in performance_rows]
    control_seconds = [as_float(row.get("control_seconds")) for row in performance_rows]
    if any(value is None for value in frame_seconds + inference_seconds + control_seconds):
        raise ValueError("Performance CSV contains a non-numeric timing value.")

    frame_values = [value for value in frame_seconds if value is not None]
    inference_values = [value for value in inference_seconds if value is not None]
    control_values = [value for value in control_seconds if value is not None]
    steady_values = frame_values[INITIAL_ROWS_TO_EXCLUDE:] or frame_values

    summary: dict[str, Any] = {
        "frames": len(performance_rows),
        "frame_seconds_total": sum(frame_values),
        "effective_fps": len(frame_values) / sum(frame_values) if sum(frame_values) else None,
        "frame_seconds_mean_steady": statistics.mean(steady_values),
        "frame_seconds_median_steady": statistics.median(steady_values),
        "frame_seconds_p95_steady": percentile(steady_values, 0.95),
        "inference_seconds_total": sum(inference_values),
        "control_seconds_total": sum(control_values),
        "gpu_energy_final_j": as_float(performance_rows[-1].get("gpu_energy_j")),
        "gpu_memory_peak_mb": max(
            value for value in (as_float(row.get("gpu_memory_used_mb")) for row in performance_rows) if value is not None
        ),
        "gpu_temperature_peak_c": max(
            value for value in (as_float(row.get("gpu_temperature_c")) for row in performance_rows) if value is not None
        ),
    }

    if not result_path.is_file():
        return summary
    with result_path.open("r", newline="", encoding="utf-8") as csv_file:
        results_by_frame = {row.get("frame"): row for row in csv.DictReader(csv_file)}

    update_results = [
        results_by_frame.get(row.get("frame"), {})
        for row in performance_rows
        if (as_float(row.get("control_seconds")) or 0.0) > 0.0
    ]
    success_values = []
    for result in update_results:
        value = str(result.get("optimizer_success", "")).strip().lower()
        if value in {"1", "true"}:
            success_values.append(1.0)
        elif value in {"0", "false"}:
            success_values.append(0.0)
    summary["optimizer_updates"] = len(success_values)
    summary["optimizer_success_rate"] = statistics.mean(success_values) if success_values else None
    return summary


def build_commands(args: argparse.Namespace, session_dir: Path, run_name: str, model: str) -> tuple[list[str], Path, Path]:
    video = project_path(args.video)
    detector_model = project_path(args.detector_model)
    predictor_checkpoint = project_path(args.predictor_checkpoint)
    if model == "mpc_herding":
        output_dir = session_dir / "mpc_herding"
        command = [
            sys.executable,
            "-m",
            "src.mpc_herding.video_flock_only",
            "--video", str(video),
            "--model", str(detector_model),
            "--output-dir", str(output_dir),
            "--run-name", run_name,
            "--confidence", "0.25",
            "--image-size", "1536",
            "--device", "0",
            "--control-period", "0.3",
            "--horizon", "3",
            "--robot-max-speed", "250",
            "--robot-start", format_coordinate(args.robot_start),
            "--max-iterations", "10",
            "--synthetic-animals", "25",
            "--target", format_coordinate(args.target),
        ]
        return command, output_dir / f"{run_name}.csv", output_dir / f"{run_name}_performance.csv"

    output_dir = session_dir / "flock_only_target_prediction"
    command = [
        sys.executable,
        "src/flock_only_target_prediction/simulate_robot_mpc_flock_target_predictor.py",
        "--video", str(video),
        "--detector-model", str(detector_model),
        "--predictor-checkpoint", str(predictor_checkpoint),
        "--robot-start-x", f"{args.robot_start[0]:g}",
        "--robot-start-y", f"{args.robot_start[1]:g}",
        "--initial-heading", "0",
        "--confidence", "0.25",
        "--image-size", "1536",
        "--predictor-device", "cuda:0",
        "--detector-device", "0",
        "--single-flock",
        "--max-speed", "250",
        "--horizon", "3",
        "--control-period", "0.3",
        "--output-dir", str(output_dir),
        "--name", run_name,
    ]
    stem = video.stem
    return (
        command,
        output_dir / run_name / f"{stem}_flock_target_mpc.csv",
        output_dir / run_name / f"{stem}_flock_target_mpc_performance.csv",
    )


def run_trial(
    args: argparse.Namespace,
    session_dir: Path,
    logs_dir: Path,
    phase: str,
    repeat: int,
    sequence: int,
    model: str,
) -> dict[str, Any]:
    run_name = f"{phase}_{repeat:02d}_{model}"
    command, result_path, performance_path = build_commands(args, session_dir, run_name, model)
    log_path = logs_dir / f"{sequence:02d}_{run_name}.log"
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(subprocess.list2cmdline(command) + "\n\n")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    wall_time = time.perf_counter() - start
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")

    row: dict[str, Any] = {
        "phase": phase,
        "repeat": repeat,
        "sequence": sequence,
        "model": model,
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": completed.returncode,
        "wall_time_seconds": wall_time,
        "command": subprocess.list2cmdline(command),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "result_csv_path": str(result_path.relative_to(PROJECT_ROOT)),
        "performance_csv_path": str(performance_path.relative_to(PROJECT_ROOT)),
        "summary_error": "",
    }
    if completed.returncode == 0:
        try:
            row.update(summarize_run(performance_path, result_path))
        except (OSError, ValueError) as error:
            row["summary_error"] = str(error)
    else:
        row["summary_error"] = f"Process exited with code {completed.returncode}."
    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_float(str(row.get(key, ""))) for row in rows]
    return [value for value in values if value is not None]


def write_measurement_summary(session_dir: Path, run_rows: list[dict[str, Any]]) -> None:
    fields = ["model", "successful_runs", "failed_runs"]
    for metric in SUMMARY_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_median", f"{metric}_min", f"{metric}_max"))

    summary_rows = []
    for model in ("mpc_herding", "flock_only_target_prediction"):
        measured = [row for row in run_rows if row["phase"] == "measurement" and row["model"] == model]
        successful = [row for row in measured if row["return_code"] == 0 and not row["summary_error"]]
        row: dict[str, Any] = {
            "model": model,
            "successful_runs": len(successful),
            "failed_runs": len(measured) - len(successful),
        }
        for metric in SUMMARY_METRICS:
            values = numeric_values(successful, metric)
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"{metric}_median"] = statistics.median(values)
                row[f"{metric}_min"] = min(values)
                row[f"{metric}_max"] = max(values)
        summary_rows.append(row)
    write_csv(session_dir / "measurement_summary.csv", summary_rows, fields)


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def write_distribution_chart(path: Path, title: str, unit: str, values_by_model: dict[str, list[float]]) -> None:
    width, height = 1000, 640
    left, right, top, bottom = 115, 55, 90, 135
    chart_width = width - left - right
    chart_height = height - top - bottom
    models = [model for model in MODEL_LABELS if values_by_model.get(model)]
    maximum = max(value for values in values_by_model.values() for value in values)
    standard_deviations = {
        model: statistics.stdev(values) if len(values) > 1 else 0.0
        for model, values in values_by_model.items()
        if values
    }
    maximum = max(
        maximum,
        *(statistics.mean(values) + standard_deviations[model] for model, values in values_by_model.items() if values),
    )
    axis_maximum = maximum * 1.15 if maximum > 0 else 1.0

    def y_position(value: float) -> float:
        return top + chart_height * (1.0 - value / axis_maximum)

    colors = {"mpc_herding": "#0f766e", "flock_only_target_prediction": "#c2410c"}
    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text { font-family: Arial, sans-serif; fill: #1f2937; } .axis { stroke: #6b7280; stroke-width: 1; } .grid { stroke: #d1d5db; stroke-width: 1; } .tick { font-size: 16px; } .label { font-size: 18px; font-weight: 600; } .title { font-size: 27px; font-weight: 700; }</style>',
        f'<text x="{width / 2}" y="42" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="68" text-anchor="middle" class="tick">Bars: mean +/- standard deviation. Dots: individual measurement runs.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>',
    ]
    for tick in range(6):
        value = axis_maximum * tick / 5
        y = y_position(value)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" class="grid"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end" class="tick">{format_value(value)}</text>')
    elements.append(f'<text x="28" y="{top + chart_height / 2}" transform="rotate(-90 28 {top + chart_height / 2})" text-anchor="middle" class="label">{html.escape(unit)}</text>')

    group_width = chart_width / len(models)
    for index, model in enumerate(models):
        values = values_by_model[model]
        mean = statistics.mean(values)
        deviation = standard_deviations[model]
        center_x = left + group_width * (index + 0.5)
        bar_width = min(190, group_width * 0.38)
        bar_top = y_position(mean)
        bar_height = top + chart_height - bar_top
        error_top = y_position(min(axis_maximum, mean + deviation))
        error_bottom = y_position(max(0.0, mean - deviation))
        elements.append(f'<rect x="{center_x - bar_width / 2:.2f}" y="{bar_top:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[model]}" opacity="0.78"/>')
        elements.append(f'<line x1="{center_x:.2f}" y1="{error_top:.2f}" x2="{center_x:.2f}" y2="{error_bottom:.2f}" stroke="#111827" stroke-width="3"/>')
        elements.append(f'<line x1="{center_x - 12:.2f}" y1="{error_top:.2f}" x2="{center_x + 12:.2f}" y2="{error_top:.2f}" stroke="#111827" stroke-width="3"/>')
        elements.append(f'<line x1="{center_x - 12:.2f}" y1="{error_bottom:.2f}" x2="{center_x + 12:.2f}" y2="{error_bottom:.2f}" stroke="#111827" stroke-width="3"/>')
        for point_index, value in enumerate(values):
            offset = (point_index - (len(values) - 1) / 2) * 13
            elements.append(f'<circle cx="{center_x + offset:.2f}" cy="{y_position(value):.2f}" r="5" fill="#111827" opacity="0.85"/>')
        elements.append(f'<text x="{center_x:.2f}" y="{top + chart_height + 36}" text-anchor="middle" class="label">{MODEL_LABELS[model]}</text>')
        elements.append(f'<text x="{center_x:.2f}" y="{top + chart_height + 62}" text-anchor="middle" class="tick">n = {len(values)} | mean = {format_value(mean)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_measurement_plots(session_dir: Path, run_rows: list[dict[str, Any]]) -> list[Path]:
    successful = [
        row
        for row in run_rows
        if row["phase"] == "measurement" and row["return_code"] == 0 and not row["summary_error"]
    ]
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_definitions = (
        ("01_wall_time.svg", "End-to-end execution time", "seconds", "wall_time_seconds"),
        ("02_effective_fps.svg", "Effective throughput", "frames per second", "effective_fps"),
        ("03_frame_latency_p95.svg", "Frame latency, p95", "seconds", "frame_seconds_p95_steady"),
        ("04_gpu_energy.svg", "GPU energy per run", "joules", "gpu_energy_final_j"),
        ("05_gpu_memory.svg", "Peak GPU memory", "MB", "gpu_memory_peak_mb"),
        ("06_gpu_temperature.svg", "Peak GPU temperature", "degrees C", "gpu_temperature_peak_c"),
        ("07_optimizer_success.svg", "MPC optimizer success rate", "proportion", "optimizer_success_rate"),
    )
    created_paths = []
    for filename, title, unit, metric in plot_definitions:
        values_by_model = {
            model: numeric_values([row for row in successful if row["model"] == model], metric)
            for model in MODEL_LABELS
        }
        if not all(values_by_model.values()):
            continue
        path = plots_dir / filename
        write_distribution_chart(path, title, unit, values_by_model)
        created_paths.append(path)
    return created_paths


def write_metadata(session_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python_executable": sys.executable,
        "warmups_per_model": args.warmups,
        "measurements_per_model": args.measurements,
        "commands": {
            "shared": {
                "video": args.video,
                "detector_model": args.detector_model,
                "confidence": 0.25,
                "image_size": 1536,
                "control_period": 0.3,
                "horizon": 3,
                "robot_start": args.robot_start,
            },
            "mpc_herding": {"max_iterations": 10, "synthetic_animals": 25, "target": args.target},
            "flock_only_target_prediction": {"predictor_checkpoint": args.predictor_checkpoint},
        },
    }
    (session_dir / "benchmark_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (session_dir / "gpu_info.txt").open("w", encoding="utf-8") as output:
        try:
            subprocess.run(["nvidia-smi"], stdout=output, stderr=subprocess.STDOUT, check=False, text=True)
        except FileNotFoundError:
            output.write("nvidia-smi is not available.\n")


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.measurements < 1:
        raise ValueError("--warmups must be non-negative and --measurements must be at least 1.")

    require_files([project_path(args.video), project_path(args.detector_model), project_path(args.predictor_checkpoint)])
    video_stem = project_path(args.video).stem
    session_name = args.session_name or f"{video_stem}_benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir = PROJECT_ROOT / "runs" / "benchmark" / session_name
    logs_dir = session_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    write_metadata(session_dir, args)

    run_rows: list[dict[str, Any]] = []
    sequence = 0
    for phase, repetitions in (("warmup", args.warmups), ("measurement", args.measurements)):
        for repeat in range(1, repetitions + 1):
            models = ("mpc_herding", "flock_only_target_prediction")
            if repeat % 2 == 0:
                models = tuple(reversed(models))
            for model in models:
                sequence += 1
                print(f"[{sequence}/{2 * (args.warmups + args.measurements)}] {phase} {repeat}: {model}", flush=True)
                row = run_trial(args, session_dir, logs_dir, phase, repeat, sequence, model)
                run_rows.append(row)
                write_csv(session_dir / "run_manifest.csv", run_rows, RUN_FIELDS)
                outcome = "OK" if row["return_code"] == 0 else f"FAILED ({row['return_code']})"
                print(f"  {outcome} in {row['wall_time_seconds']:.2f} s", flush=True)

    write_measurement_summary(session_dir, run_rows)
    plot_paths = write_measurement_plots(session_dir, run_rows)
    failed_runs = [row for row in run_rows if row["return_code"] != 0 or row["summary_error"]]
    print(f"Results: {session_dir}", flush=True)
    if plot_paths:
        print(f"Plots: {session_dir / 'plots'}", flush=True)
    return 1 if failed_runs else 0


if __name__ == "__main__":
    raise SystemExit(main())
