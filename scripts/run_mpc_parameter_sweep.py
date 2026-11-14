"""Benchmark the MPC video pipeline across control and herd-size parameters.

The measurement unit is a processed frame.  Warm-up trials are intentionally
kept out of every output table so model loading and CUDA compilation do not
distort the reported distributions.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
METRICS = ("inference_seconds", "control_seconds", "frame_seconds")
BASELINE_PIPELINE_ARGS = (
    "--horizon", "3",
    "--max-iterations", "10",
    "--guidance-mode", "adaptive-sweep",
)
VIDEO_TARGETS = {
    "rebano_01": "80,1000",
    "rebano_02": "1840,600",
    "rebano_03": "1000,80",
    "rebano_04": "1840,600",
    "rebano_05": "400,80",
    "rebano_06": "1000,80",
    "rebano_07": "1000,80",
    "rebano_08": "600,80",
    "rebano_09": "80,520",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure every direct video in a folder for an MPC parameter grid."
    )
    parser.add_argument("--videos-dir", default="videos", help="Only direct video files are included.")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Video filenames, project-relative paths, or absolute paths. Defaults to all direct videos in --videos-dir.",
    )
    parser.add_argument(
        "--default-target",
        default="1840,600",
        metavar="X,Y",
        help="Target for videos without a configured rebano_01...09 target (default: 1840,600).",
    )
    parser.add_argument(
        "--video-target",
        action="append",
        default=[],
        metavar="PATH=X,Y",
        help="Per-video target. Repeat this option as needed; overrides configured and default targets.",
    )
    parser.add_argument("--model", default="models/dogRobot_v2_4_2_best.pt")
    parser.add_argument("--output-root", default="runs/mpc_parameter_sweep")
    parser.add_argument("--session-name", default=None)
    parser.add_argument(
        "--regenerate-plots",
        metavar="SESSION_DIR",
        default=None,
        help="Regenerate plots from an existing sweep session without rerunning videos.",
    )
    parser.add_argument("--control-interval-frames", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--synthetic-animals", nargs="+", type=int, default=[10, 50, 120])
    parser.add_argument(
        "--parameter-grid",
        nargs="+",
        default=None,
        metavar="OPTION=VALUES",
        help=(
            "Replace the default grid with arbitrary video_flock_only options. "
            "Example: --parameter-grid control-interval-frames=1,5,10 "
            "synthetic-animals=10,25,50,120 horizon=3,5. Use | instead of , "
            "when a value itself contains commas, such as robot-start=80,80|120,80."
        ),
    )
    parser.add_argument("--warmups", type=int, default=1, help="Unreported warm-up runs per video and combination.")
    parser.add_argument(
        "--no-warmups",
        dest="warmups",
        action="store_const",
        const=0,
        help="Disable warm-up runs; equivalent to --warmups 0.",
    )
    parser.add_argument("--measurements", type=int, default=3, help="Reported repeated runs per video and combination.")
    parser.add_argument("--exclude-initial-frames", type=int, default=30, help="Frames excluded from every measured run before aggregation.")
    parser.add_argument("--seed", type=int, default=20261113, help="Seed used to randomize parameter combinations.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--fixed-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra video_flock_only arguments shared by every run; place this option last.",
    )
    return parser.parse_args()


def require_valid_args(args: argparse.Namespace) -> None:
    if args.warmups < 0 or args.measurements < 1 or args.exclude_initial_frames < 0:
        raise ValueError("--warmups and --exclude-initial-frames must be non-negative; --measurements must be at least 1.")
    if any(value < 1 for value in args.control_interval_frames):
        raise ValueError("Every --control-interval-frames value must be at least 1.")
    if any(value < 1 for value in args.synthetic_animals):
        raise ValueError("Every --synthetic-animals value must be at least 1.")
    forbidden = {"--video", "--model", "--output-dir", "--run-name", "--performance-report", "--control-interval-frames", "--synthetic-animals", "--target"}
    supplied = {value.split("=", maxsplit=1)[0] for value in args.fixed_args}
    conflict = sorted(forbidden & supplied)
    if conflict:
        raise ValueError(f"--fixed-args cannot override benchmark-controlled options: {', '.join(conflict)}")


def parameter_field(option: str) -> str:
    return option.removeprefix("--").replace("-", "_")


def parameter_abbreviation(field: str) -> str:
    return "".join(part[0] for part in field.split("_") if part).upper()


def parse_parameter_grid(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.parameter_grid is None:
        return [
            ("--control-interval-frames", [str(value) for value in sorted(set(args.control_interval_frames))]),
            ("--synthetic-animals", [str(value) for value in sorted(set(args.synthetic_animals))]),
        ]

    protected = {"--video", "--model", "--output-dir", "--run-name", "--performance-report", "--target"}
    grid: list[tuple[str, list[str]]] = []
    fields: set[str] = set()
    for definition in args.parameter_grid:
        if "=" not in definition:
            raise ValueError(f"Invalid grid definition '{definition}'. Use option=value1,value2.")
        option, values_text = definition.split("=", maxsplit=1)
        option = f"--{option.lstrip('-')}"
        separator = "|" if "|" in values_text else ","
        values = [value.strip() for value in values_text.split(separator) if value.strip()]
        field = parameter_field(option)
        if option in protected or not values or field in fields:
            raise ValueError(f"Invalid grid option '{definition}'. Options must be unique, configurable and have one or more values.")
        fields.add(field)
        grid.append((option, values))
    return grid


def direct_videos(videos_dir: Path) -> list[Path]:
    return sorted(
        (path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def selected_videos(videos_dir: Path, requested_videos: list[str] | None) -> list[Path]:
    if requested_videos is None:
        return direct_videos(videos_dir)

    videos: list[Path] = []
    for requested_path in requested_videos:
        video = resolve_requested_video(videos_dir, requested_path)
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            raise FileNotFoundError(f"Requested video not found or unsupported: {requested_path}")
        video = video.resolve()
        if video not in videos:
            videos.append(video)
    return videos


def resolve_requested_video(videos_dir: Path, requested_path: str) -> Path:
    requested = Path(requested_path)
    return videos_dir / requested if requested.parent == Path(".") else project_path(requested)


def valid_target(target: str) -> bool:
    try:
        x_text, y_text = target.split(",", maxsplit=1)
        float(x_text)
        float(y_text)
        return True
    except ValueError:
        return False


def parse_video_targets(videos_dir: Path, videos: list[Path], definitions: list[str]) -> dict[Path, str]:
    selected = {video.resolve() for video in videos}
    targets: dict[Path, str] = {}
    for definition in definitions:
        if "=" not in definition:
            raise ValueError(f"Invalid --video-target '{definition}'. Use PATH=X,Y.")
        path_text, target = definition.rsplit("=", maxsplit=1)
        video = resolve_requested_video(videos_dir, path_text).resolve()
        if video not in selected:
            raise ValueError(f"--video-target references a video not selected with --videos: {path_text}")
        if not valid_target(target):
            raise ValueError(f"Invalid target '{target}'. Use X,Y.")
        targets[video] = target
    return targets


def target_for_video(video: Path, default_target: str, explicit_targets: dict[Path, str]) -> str:
    target = explicit_targets.get(video.resolve(), VIDEO_TARGETS.get(video.stem.lower(), default_target))
    if target is None:
        raise ValueError(
            f"No target configured for video: {video.name}. Use --default-target X,Y for this video."
        )
    return target


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def combination_key(row: dict[str, Any], parameter_fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in parameter_fields)


def combination_label(row: dict[str, Any], parameter_fields: list[str]) -> str:
    field_labels = {"control_interval_frames": "I", "synthetic_animals": "N"}
    return "\n".join(f"{field_labels.get(field, field)}={row[field]}" for field in parameter_fields)


def best_combination_text(
    rows: list[dict[str, Any]],
    parameter_fields: list[str],
    metric: str,
    higher_is_better: bool,
    multiplier: float,
    unit: str,
) -> str:
    candidates = [row for row in rows if as_float(row.get(metric)) is not None]
    if not candidates:
        return "No hay datos suficientes para seleccionar una combinacion."
    best = (max if higher_is_better else min)(candidates, key=lambda row: float(row[metric]))
    value = float(best[metric]) * multiplier
    label = combination_label(best, parameter_fields).replace("\n", ", ")
    criterion = "maximo" if higher_is_better else "minimo"
    return f"Mejor combinacion segun el {criterion} de esta metrica: {label} ({value:.2f} {unit})."


def best_combination_by_video_text(
    rows: list[dict[str, Any]],
    parameter_fields: list[str],
    metric: str,
    higher_is_better: bool,
    multiplier: float,
    unit: str,
) -> str:
    fragments = []
    for video in sorted({str(row["video"]) for row in rows}):
        candidates = [row for row in rows if row["video"] == video and as_float(row.get(metric)) is not None]
        if not candidates:
            continue
        best = (max if higher_is_better else min)(candidates, key=lambda row: float(row[metric]))
        label = combination_label(best, parameter_fields).replace("\n", ", ")
        fragments.append(f"{video}: {label} ({float(best[metric]) * multiplier:.2f} {unit})")
    criterion = "maximo" if higher_is_better else "minimo"
    return f"Mejor combinacion por video segun el {criterion}: " + "; ".join(fragments) + "."


def read_video_fps(videos: list[Path]) -> dict[str, float]:
    try:
        import cv2
    except ImportError:
        return {}
    fps_by_video = {}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if fps > 0:
            fps_by_video[video.name] = fps
    return fps_by_video


def build_feasibility_summary(
    raw_rows: list[dict[str, Any]],
    parameter_fields: list[str],
    fps_by_video: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = {}
    deadlines: dict[tuple[str, ...], list[bool]] = {}
    for row in raw_rows:
        key = combination_key(row, parameter_fields)
        values = grouped.setdefault(key, {"frame": [], "control_update": []})
        frame_seconds = as_float(row["frame_seconds"])
        control_seconds = as_float(row["control_seconds"])
        if frame_seconds is not None:
            values["frame"].append(frame_seconds)
            fps = fps_by_video.get(str(row["video"]))
            if fps is not None:
                deadlines.setdefault(key, []).append(frame_seconds <= 1.0 / fps)
        if control_seconds is not None and control_seconds > 0.0:
            values["control_update"].append(control_seconds)

    output = []
    for key, values in sorted(grouped.items()):
        frame_values = values["frame"]
        updates = values["control_update"]
        deadline_values = deadlines.get(key, [])
        output.append(
            {
                **dict(zip(parameter_fields, key)),
                "measured_frames": len(frame_values),
                "frame_seconds_p95": percentile(frame_values, 0.95),
                "frame_seconds_p99": percentile(frame_values, 0.99),
                "deadline_frames": len(deadline_values),
                "deadline_compliance_pct": 100.0 * sum(deadline_values) / len(deadline_values) if deadline_values else "",
                "mpc_updates": len(updates),
                "control_update_seconds_mean": statistics.mean(updates) if updates else "",
                "control_update_seconds_median": statistics.median(updates) if updates else "",
                "control_update_seconds_p95": percentile(updates, 0.95) if updates else "",
            }
        )
    return output


def build_control_reliability_summary(session_dir: Path, parameter_fields: list[str]) -> list[dict[str, Any]]:
    manifest_path = session_dir / "run_manifest.csv"
    if not manifest_path.is_file():
        return []
    grouped: dict[tuple[str, ...], dict[str, int]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as source:
        runs = list(csv.DictReader(source))
    for run in runs:
        if run.get("phase") != "measurement" or run.get("return_code") != "0":
            continue
        performance_path = session_dir / Path(run["performance_csv"])
        result_path = performance_path.with_name(performance_path.name.removesuffix("_performance.csv") + ".csv")
        if not performance_path.is_file() or not result_path.is_file():
            continue
        with result_path.open("r", newline="", encoding="utf-8") as source:
            result_by_frame = {row["frame"]: row for row in csv.DictReader(source)}
        with performance_path.open("r", newline="", encoding="utf-8") as source:
            performance_rows = list(csv.DictReader(source))
        key = combination_key(run, parameter_fields)
        counts = grouped.setdefault(key, {"mpc_updates": 0, "optimizer_successes": 0, "control_recoveries": 0})
        for performance_row in performance_rows:
            if (as_float(performance_row.get("control_seconds")) or 0.0) <= 0.0:
                continue
            result = result_by_frame.get(performance_row["frame"], {})
            counts["mpc_updates"] += 1
            counts["optimizer_successes"] += str(result.get("optimizer_success", "")).lower() in {"1", "true"}
            counts["control_recoveries"] += str(result.get("control_recovery", "")).lower() in {"1", "true"}
    return [
        {
            **dict(zip(parameter_fields, key)),
            **counts,
            "optimizer_success_pct": 100.0 * counts["optimizer_successes"] / counts["mpc_updates"] if counts["mpc_updates"] else 0.0,
            "control_recovery_pct": 100.0 * counts["control_recoveries"] / counts["mpc_updates"] if counts["mpc_updates"] else 0.0,
        }
        for key, counts in sorted(grouped.items())
    ]


def enrich_combination_summary(
    summary_rows: list[dict[str, Any]],
    feasibility_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    parameter_fields: list[str],
) -> None:
    feasibility_by_key = {combination_key(row, parameter_fields): row for row in feasibility_rows}
    reliability_by_key = {combination_key(row, parameter_fields): row for row in reliability_rows}
    for row in summary_rows:
        key = combination_key(row, parameter_fields)
        feasibility = feasibility_by_key.get(key, {})
        reliability = reliability_by_key.get(key, {})
        row.update(
            {
                "frame_seconds_p95": feasibility.get("frame_seconds_p95", ""),
                "frame_seconds_p99": feasibility.get("frame_seconds_p99", ""),
                "deadline_frames": feasibility.get("deadline_frames", ""),
                "deadline_compliance_pct": feasibility.get("deadline_compliance_pct", ""),
                "mpc_updates_measured_frames": feasibility.get("mpc_updates", ""),
                "control_update_seconds_mean": feasibility.get("control_update_seconds_mean", ""),
                "control_update_seconds_median": feasibility.get("control_update_seconds_median", ""),
                "control_update_seconds_p95": feasibility.get("control_update_seconds_p95", ""),
                "mpc_updates_all_frames": reliability.get("mpc_updates", ""),
                "optimizer_successes": reliability.get("optimizer_successes", ""),
                "control_recoveries": reliability.get("control_recoveries", ""),
                "optimizer_success_pct": reliability.get("optimizer_success_pct", ""),
                "control_recovery_pct": reliability.get("control_recovery_pct", ""),
            }
        )


def sort_combination_keys(keys: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    def component(value: str) -> tuple[int, float | str]:
        try:
            return 0, float(value)
        except ValueError:
            return 1, value

    return sorted(keys, key=lambda key: tuple(component(value) for value in key))


def build_per_video_summary(
    raw_rows: list[dict[str, Any]],
    parameter_fields: list[str],
    fps_by_video: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, list[float]]] = {}
    deadline_values: dict[tuple[str, tuple[str, ...]], list[bool]] = {}
    for row in raw_rows:
        key = (str(row["video"]), combination_key(row, parameter_fields))
        values = grouped.setdefault(key, {"frame": [], "inference": [], "control_update": []})
        for source, target in (("frame_seconds", "frame"), ("inference_seconds", "inference")):
            value = as_float(row[source])
            if value is not None:
                values[target].append(value)
        control_seconds = as_float(row["control_seconds"])
        if control_seconds is not None and control_seconds > 0.0:
            values["control_update"].append(control_seconds)
        frame_seconds = as_float(row["frame_seconds"])
        fps = fps_by_video.get(str(row["video"]))
        if frame_seconds is not None and fps is not None:
            deadline_values.setdefault(key, []).append(frame_seconds <= 1.0 / fps)

    output = []
    for (video, key), values in sorted(grouped.items(), key=lambda item: (item[0][0], sort_combination_keys([item[0][1]])[0])):
        frames = values["frame"]
        inference = values["inference"]
        updates = values["control_update"]
        deadlines = deadline_values.get((video, key), [])
        output.append(
            {
                "video": video,
                **dict(zip(parameter_fields, key)),
                "measured_frames": len(frames),
                "frame_seconds_mean": statistics.mean(frames),
                "frame_seconds_p95": percentile(frames, 0.95),
                "inference_seconds_mean": statistics.mean(inference),
                "control_update_seconds_median": statistics.median(updates) if updates else "",
                "effective_fps": 1.0 / statistics.mean(frames) if frames else 0.0,
                "deadline_compliance_pct": 100.0 * sum(deadlines) / len(deadlines) if deadlines else "",
            }
        )
    return output


def create_by_video_report(
    output_path: Path,
    plots: list[tuple[Path, str, str]],
    scene_images: list[tuple[Path, str]],
) -> bool:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError:
        print("Word report not generated: install python-docx.", flush=True)
        return False

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    normal = document.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    for index, (plot_path, title, explanation) in enumerate(plots):
        if index:
            document.add_page_break()
        document.add_heading(title, level=1)
        document.add_picture(str(plot_path), width=Inches(6.9))
        document.add_heading("Interpretacion", level=2)
        document.add_paragraph(explanation)
    document.add_page_break()
    document.add_heading("Conclusion de la comparacion entre videos", level=1)
    document.add_paragraph(
        "Las ocho graficas muestran que el contenido del video introduce variabilidad medible en la latencia, "
        "especialmente en las configuraciones de menor coste. Por ejemplo, con I=10 y N=10, la latencia media "
        "por video varia aproximadamente entre 58.7 y 115.9 ms. Al aumentar el numero de animales sinteticos, "
        "el tiempo de MPC pasa a dominar y la variacion relativa entre videos disminuye."
    )
    document.add_paragraph(
        "Las capturas de las escenas ayudan a contextualizar esta variabilidad: rebano_01 y rebano_02 contienen "
        "rebanos densos y sombras extensas; rebano_03 incorpora estructuras y posibles oclusiones; rebano_09 "
        "muestra un rebano mas disperso sobre vegetacion. Estos factores son compatibles con cambios en la carga "
        "del detector, pero este experimento no permite establecer causalidad porque solo hay una ejecucion por "
        "video y combinacion."
    )
    document.add_paragraph(
        "La conclusion defendible es computacional: los parametros del MPC tienen un efecto mayor que el video "
        "en las configuraciones costosas, y ninguna combinacion medida satisface de forma sostenida el plazo de "
        "un video cercano a 30 FPS. Estas medidas no demuestran eficacia de pastoreo, ya que el rebano grabado "
        "no responde al perro simulado."
    )
    for scene_path, title in scene_images:
        document.add_page_break()
        document.add_heading(title, level=1)
        document.add_picture(str(scene_path), width=Inches(6.9))
        document.add_paragraph("Captura anotada representativa del video utilizada para contextualizar las mediciones de rendimiento.")
    document.save(output_path)
    return True


def create_global_plots_report(
    plots_dir: Path,
    summary_rows: list[dict[str, Any]],
    parameter_fields: list[str],
) -> bool:
    definitions = (
        ("boxplot_inference_seconds.png", "Distribucion de latencia de inferencia YOLO", "Cada caja resume la distribucion por frame de la latencia de YOLO para una combinacion. La linea central es la mediana, la caja contiene el intervalo intercuartil y los puntos muestran valores atipicos.", "inference_seconds_mean", False, 1_000.0, "ms"),
        ("boxplot_control_seconds.png", "Distribucion de coste MPC amortizado por frame", "Incluye los frames sin actualizacion MPC, cuyo coste de control es cero. Por tanto, describe el coste amortizado por frame, no el tiempo de una optimizacion individual.", "control_seconds_mean", False, 1_000.0, "ms"),
        ("boxplot_frame_seconds.png", "Distribucion de latencia total por frame", "Representa el tiempo extremo a extremo por frame, incluyendo inferencia, MPC, anotacion y escritura de video. Es la medida principal de rendimiento del pipeline completo.", "frame_seconds_mean", False, 1_000.0, "ms"),
        ("statistics_inference_seconds.png", "Media, mediana y desviacion de inferencia YOLO", "Las barras comparan tendencia central y variabilidad de la inferencia. Las diferencias entre combinaciones reflejan principalmente variabilidad de contenido y carga del sistema, ya que los animales sinteticos no cambian la entrada de YOLO.", "inference_seconds_mean", False, 1_000.0, "ms"),
        ("statistics_control_seconds.png", "Media, mediana y desviacion del coste MPC", "Resume el coste MPC amortizado por frame. Las medianas iguales a cero en intervalos altos indican que muchos frames mantienen el ultimo comando sin resolver una nueva optimizacion.", "control_seconds_mean", False, 1_000.0, "ms"),
        ("statistics_frame_seconds.png", "Media, mediana y desviacion de la latencia total", "Compara el coste extremo a extremo de cada configuracion. La desviacion tipica muestra la estabilidad temporal y complementa la media y mediana.", "frame_seconds_mean", False, 1_000.0, "ms"),
        ("deadline_compliance_heatmap.png", "Cumplimiento del plazo temporal del video", "Cada celda indica el porcentaje de frames cuya latencia total no supera el plazo 1/FPS del video original. Es la medida directa de viabilidad en tiempo real.", "deadline_compliance_pct", True, 1.0, "%"),
        ("frame_latency_ecdf.png", "Funcion de distribucion acumulada de latencia", "Cada curva muestra la proporcion de frames procesados por debajo de cada latencia. Permite comparar mediana, colas de latencia y percentiles sin ocultar la forma de la distribucion.", "frame_seconds_p95", False, 1_000.0, "ms"),
        ("latency_fps_pareto.png", "Compromiso entre p95 de latencia y FPS", "Cada punto representa una configuracion. Menor p95 y mayor FPS son preferibles; la figura ayuda a identificar configuraciones dominadas y candidatas operativas.", "frame_seconds_p95", False, 1_000.0, "ms"),
        ("mpc_optimizer_reliability.png", "Fiabilidad del optimizador MPC", "Muestra el porcentaje de actualizaciones en las que el optimizador informa exito y la frecuencia con la que se activa la recuperacion de control. Evalua estabilidad numerica, no eficacia de pastoreo.", "optimizer_success_pct", True, 1.0, "%"),
        ("mpc_update_latency_boxplot.png", "Latencia de las actualizaciones MPC", "Solo incluye frames que ejecutan una optimizacion MPC. Es la grafica adecuada para medir el tiempo real de resolucion, sin mezclar frames que conservan el control anterior.", "control_update_seconds_median", False, 1_000.0, "ms"),
    )
    available = [
        (
            plots_dir / filename,
            title,
            f"{explanation}\n\n{best_combination_text(summary_rows, parameter_fields, metric, higher_is_better, multiplier, unit)}",
        )
        for filename, title, explanation, metric, higher_is_better, multiplier, unit in definitions
        if (plots_dir / filename).is_file()
    ]
    if not available:
        return False
    try:
        from docx import Document
        from docx.shared import Inches, Pt
    except ImportError:
        print("Global Word report not generated: install python-docx.", flush=True)
        return False
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)
    for index, (plot_path, title, explanation) in enumerate(available):
        if index:
            document.add_page_break()
        document.add_heading(title, level=1)
        document.add_picture(str(plot_path), width=Inches(6.9))
        document.add_heading("Interpretacion", level=2)
        document.add_paragraph(explanation)
    document.add_page_break()
    document.add_heading("Conclusion de las mediciones globales", level=1)
    document.add_paragraph(
        "El numero de animales sinteticos y la frecuencia de actualización del MPC determinan el coste principal "
        "del pipeline. Las configuraciones con N=10 presentan las latencias más bajas, mientras que incrementar "
        "N a 50 y 120 eleva de forma marcada el tiempo de optimización y reduce los FPS efectivos."
    )
    document.add_paragraph(
        "Para el criterio estrictamente computacional utilizado aquí, la combinación I=10, N=10 es la mejor "
        "candidata operativa: minimiza la latencia global y maximiza el rendimiento efectivo entre las "
        "configuraciones evaluadas. Sin embargo, incluso esta configuración no cumple de forma sostenida el "
        "plazo de un vídeo cercano a 30 FPS, por lo que el pipeline no puede considerarse de tiempo real en "
        "estas condiciones de ejecución."
    )
    document.add_paragraph(
        "La tasa de éxito y las recuperaciones del optimizador informan sobre estabilidad numérica, no sobre "
        "eficacia de pastoreo. Para afirmar que el controlador guía mejor al rebaño se necesitaría un experimento "
        "en lazo cerrado, en el que el rebaño responda al perro simulado o al robot real."
    )
    document.save(plots_dir / "informe_mediciones_globales.docx")
    return True


def extract_representative_frames(session_dir: Path, output_dir: Path) -> list[tuple[Path, str]]:
    try:
        import cv2
    except ImportError:
        print("Scene captures not generated: install OpenCV.", flush=True)
        return []
    manifest_path = session_dir / "run_manifest.csv"
    if not manifest_path.is_file():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as source:
        runs = list(csv.DictReader(source))
    preferred = [
        run
        for run in runs
        if run.get("phase") == "measurement"
        and run.get("return_code") == "0"
        and run.get("control_interval_frames") == "10"
        and run.get("synthetic_animals") == "10"
    ]
    runs_by_video = {run["video"]: run for run in preferred}
    requested_frames = {"rebano_01": 630, "rebano_02": 72, "rebano_03": 206}
    scenes: list[tuple[Path, str]] = []
    for video_name in sorted(runs_by_video):
        run = runs_by_video[video_name]
        performance_path = session_dir / Path(run["performance_csv"])
        video_path = performance_path.with_name(performance_path.name.removesuffix("_performance.csv") + ".mp4")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            continue
        stem = Path(video_name).stem.lower()
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = min(requested_frames.get(stem, max(0, frame_count // 2)), max(0, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        capture.release()
        if not success:
            continue
        output_path = output_dir / f"scene_{stem}_frame_{frame_index}.png"
        cv2.imwrite(str(output_path), frame)
        scenes.append((output_path, f"Escena representativa: {video_name} (frame {frame_index})"))
    return scenes


def create_by_video_plots(
    plots_dir: Path,
    per_video_rows: list[dict[str, Any]],
    parameter_fields: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Per-video plots not generated: install matplotlib.", flush=True)
        return []
    if not per_video_rows:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted({str(row["video"]) for row in per_video_rows})
    keys = sort_combination_keys(list({combination_key(row, parameter_fields) for row in per_video_rows}))
    labels = [combination_label(dict(zip(parameter_fields, key)), parameter_fields).replace("\n", " ") for key in keys]
    index = {(str(row["video"]), combination_key(row, parameter_fields)): row for row in per_video_rows}
    definitions = (
        ("01_frame_latency_mean_heatmap.png", "Mean total frame latency by video", "frame_seconds_mean", 1_000.0, "milliseconds", "viridis", False, "Cada celda es el tiempo medio de procesamiento extremo a extremo por frame. Los valores bajos son preferibles; la variación entre columnas cuantifica el efecto del contenido del vídeo sobre el tiempo total."),
        ("02_frame_latency_p95_heatmap.png", "p95 total frame latency by video", "frame_seconds_p95", 1_000.0, "milliseconds", "magma", False, "Cada celda es el percentil 95 de la latencia total por frame. Caracteriza el comportamiento de cola sin depender del 5% de frames más lentos."),
        ("03_inference_latency_mean_heatmap.png", "Mean YOLO inference latency by video", "inference_seconds_mean", 1_000.0, "milliseconds", "viridis", False, "Cada celda es la latencia media del detector. Las diferencias entre vídeos reflejan contenido visual y carga del detector; los animales sintéticos no modifican directamente la entrada de YOLO."),
        ("04_mpc_update_latency_median_heatmap.png", "Median MPC update latency by video", "control_update_seconds_median", 1_000.0, "milliseconds", "magma", False, "Solo se incluyen frames que ejecutan una optimización MPC. Así se mide el tiempo real de resolución sin mezclar frames que conservan el comando anterior."),
        ("05_effective_fps_heatmap.png", "Effective throughput by video", "effective_fps", 1.0, "frames per second", "viridis", True, "Los FPS efectivos son la inversa de la latencia media total por frame. Los valores altos son preferibles y pueden compararse con los FPS originales del vídeo para evaluar viabilidad en tiempo real."),
        ("06_deadline_compliance_heatmap.png", "Video-deadline compliance by video", "deadline_compliance_pct", 1.0, "percent", "RdYlGn", True, "Cada celda es el porcentaje de frames procesados dentro del plazo 1/FPS de ese vídeo. Un valor próximo a 100% indica procesamiento en tiempo real; un valor bajo indica que el pipeline no puede mantener el ritmo."),
    )
    report_plots: list[tuple[Path, str, str]] = []
    paths: list[Path] = []
    for filename, title, metric, multiplier, unit, cmap, higher_is_better, explanation in definitions:
        matrix = []
        for key in keys:
            values = []
            for video in videos:
                value = index[(video, key)].get(metric, "")
                values.append(float(value) * multiplier if value != "" else float("nan"))
            matrix.append(values)
        figure, axis = plt.subplots(figsize=(max(8, len(videos) * 2.0), max(6, len(labels) * 0.52)), layout="constrained")
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set(title=title, xlabel="video", ylabel="parameter combination")
        axis.set_xticks(range(len(videos)), videos, rotation=25, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                text = f"{value:.1f}" if abs(value) >= 1.0 else f"{value:.2f}"
                axis.text(column_index, row_index, text, ha="center", va="center", color="black" if image.norm(value) > 0.55 else "white", fontsize=8)
        figure.colorbar(image, ax=axis, label=unit)
        path = plots_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
        report_plots.append((path, title, f"{explanation}\n\n{best_combination_by_video_text(per_video_rows, parameter_fields, metric, higher_is_better, multiplier, unit)}"))

    if {"control_interval_frames", "synthetic_animals"}.issubset(parameter_fields):
        intervals = sorted({str(row["control_interval_frames"]) for row in per_video_rows}, key=float)
        animals = sorted({str(row["synthetic_animals"]) for row in per_video_rows}, key=float)
        line_definitions = (
            (
                "07_frame_latency_p95_lines.png",
                "p95 total frame latency: video comparison",
                "frame_seconds_p95",
                1_000.0,
                "p95 total frame latency (ms)",
                "Cada panel fija el intervalo de control. Las líneas comparan los vídeos mientras aumenta el número de animales sintéticos; el eje vertical logarítmico permite visualizar configuraciones rápidas y lentas a la vez.",
                False,
            ),
            (
                "08_mpc_update_latency_lines.png",
                "Median MPC update latency: video comparison",
                "control_update_seconds_median",
                1_000.0,
                "median MPC update latency (ms)",
                "Solo se representan frames con actualización MPC. La gráfica aísla cómo el tamaño del rebaño sintético y el contenido del vídeo afectan al tiempo necesario para optimizar.",
                False,
            ),
        )
        for filename, title, metric, multiplier, ylabel, explanation, higher_is_better in line_definitions:
            figure, axes = plt.subplots(1, len(intervals), figsize=(max(11, len(intervals) * 4.2), 5.8), sharey=True, layout="constrained")
            if len(intervals) == 1:
                axes = [axes]
            for axis, interval in zip(axes, intervals):
                for video in videos:
                    values = [
                        float(
                            index[
                                (
                                    video,
                                    tuple(
                                        {"control_interval_frames": interval, "synthetic_animals": animal}[field]
                                        for field in parameter_fields
                                    ),
                                )
                            ][metric]
                        ) * multiplier
                        for animal in animals
                    ]
                    axis.plot([float(animal) for animal in animals], values, marker="o", linewidth=2, label=video)
                axis.set(title=f"I = {interval}", xlabel="synthetic animals", yscale="log")
                axis.grid(which="both", alpha=0.3)
            axes[0].set_ylabel(ylabel)
            handles, legend_labels = axes[-1].get_legend_handles_labels()
            figure.legend(handles, legend_labels, title="video", loc="upper center", ncols=len(videos))
            path = plots_dir / filename
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)
            report_plots.append((path, title, f"{explanation}\n\n{best_combination_by_video_text(per_video_rows, parameter_fields, metric, higher_is_better, multiplier, 'milliseconds')}"))
    scene_images = extract_representative_frames(plots_dir.parents[1], plots_dir)
    create_by_video_report(plots_dir / "informe_mediciones_por_video_con_conclusion.docx", report_plots, scene_images)
    return paths


def build_command(
    args: argparse.Namespace,
    video: Path,
    output_dir: Path,
    run_name: str,
    combination: dict[str, str],
    grid: list[tuple[str, list[str]]],
    target: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.mpc_herding.video_flock_only",
        "--video", str(video),
        "--model", str(project_path(args.model)),
        "--output-dir", str(output_dir),
        "--run-name", run_name,
        "--performance-report", "frame",
        "--device", str(args.device),
        "--image-size", str(args.image_size),
        "--confidence", str(args.confidence),
        "--iou", str(args.iou),
        "--target", target,
        "--progress-interval", "1000",
        *BASELINE_PIPELINE_ARGS,
    ]
    for option, _ in grid:
        value = combination[parameter_field(option)]
        if value.lower() == "true":
            command.append(option)
        elif value.lower() != "false":
            command.extend((option, value))
    return [*command, *args.fixed_args]


def append_frame_rows(
    raw_rows: list[dict[str, Any]],
    performance_path: Path,
    run: dict[str, Any],
    exclude_initial_frames: int,
    parameter_fields: list[str],
) -> int:
    with performance_path.open("r", newline="", encoding="utf-8") as source:
        performance_rows = list(csv.DictReader(source))
    for row in performance_rows:
        frame = int(row["frame"])
        if frame < exclude_initial_frames:
            continue
        raw_rows.append(
            {
                "video": run["video"],
                **{field: run[field] for field in parameter_fields},
                "repeat": run["repeat"],
                "frame": frame,
                **{metric: row.get(metric, "") for metric in METRICS},
            }
        )
    return len(performance_rows)


def aggregate_by_frame(raw_rows: list[dict[str, Any]], parameter_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in raw_rows:
        key = (row["video"], *(row[field] for field in parameter_fields), row["frame"])
        values = grouped.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = as_float(row[metric])
            if value is not None:
                values[metric].append(value)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row: dict[str, Any] = {
            "video": key[0], **dict(zip(parameter_fields, key[1:-1])), "frame": key[-1],
            "measurement_repeats": len(values["frame_seconds"]),
        }
        for metric, samples in values.items():
            for name, value in stats(samples).items():
                row[f"{metric}_{name}"] = value
        output.append(row)
    return output


def aggregate_combinations(raw_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], parameter_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    for row in raw_rows:
        key = tuple(row[field] for field in parameter_fields)
        values = grouped.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            value = as_float(row[metric])
            if value is not None:
                values[metric].append(value)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        successful = [
            run for run in run_rows
            if run["phase"] == "measurement" and run["return_code"] == 0
            and all(run[field] == value for field, value in zip(parameter_fields, key))
        ]
        row: dict[str, Any] = {
            **dict(zip(parameter_fields, key)),
            "videos": len({run["video"] for run in successful}),
            "successful_measurement_runs": len(successful),
            "measured_frames": len(values["frame_seconds"]),
        }
        for metric, samples in values.items():
            for name, value in stats(samples).items():
                row[f"{metric}_{name}"] = value
        frame_mean = row["frame_seconds_mean"]
        row["effective_fps_from_mean_frame"] = 1.0 / frame_mean if frame_mean else 0.0
        output.append(row)
    return output


def create_plots(
    plots_dir: Path,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    parameter_fields: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots not generated: install matplotlib.", flush=True)
        return []

    plots_dir.mkdir(exist_ok=True)
    field_labels = {
        "control_interval_frames": "I",
        "synthetic_animals": "N",
    }
    labels = [
        "\n".join(f"{field_labels.get(field, field)}={row[field]}" for field in parameter_fields)
        for row in summary_rows
    ]
    paths: list[Path] = []
    for metric, title in (("frame_seconds", "Total frame latency"), ("inference_seconds", "YOLO inference latency"), ("control_seconds", "MPC control latency")):
        values = [
            [as_float(raw[metric]) for raw in raw_rows if all(raw[field] == row[field] for field in parameter_fields)]
            for row in summary_rows
        ]
        values = [[value * 1000.0 for value in samples if value is not None] for samples in values]
        figure, axis = plt.subplots(figsize=(max(12, len(labels) * 1.15), 7), layout="constrained")
        axis.boxplot(values, tick_labels=labels, showfliers=True)
        axis.set(title=f"{title} by parameter combination", ylabel="milliseconds", xlabel="parameter combination")
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.3)
        path = plots_dir / f"boxplot_{metric}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

        means = [row[f"{metric}_mean"] * 1000.0 for row in summary_rows]
        medians = [row[f"{metric}_median"] * 1000.0 for row in summary_rows]
        deviations = [row[f"{metric}_std"] * 1000.0 for row in summary_rows]
        positions = list(range(len(labels)))
        figure, axis = plt.subplots(figsize=(max(12, len(labels) * 1.15), 7), layout="constrained")
        axis.bar([position - 0.2 for position in positions], means, 0.4, yerr=deviations, capsize=3, label="mean +/- standard deviation")
        axis.bar([position + 0.2 for position in positions], medians, 0.4, label="median")
        axis.set(title=f"{title}: mean, median and standard deviation", ylabel="milliseconds", xlabel="parameter combination")
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        path = plots_dir / f"statistics_{metric}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def create_feasibility_plots(
    plots_dir: Path,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    feasibility_rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    parameter_fields: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Feasibility plots not generated: install matplotlib.", flush=True)
        return []

    plots_dir.mkdir(exist_ok=True)
    labels = [combination_label(row, parameter_fields) for row in summary_rows]
    by_key = {combination_key(row, parameter_fields): row for row in feasibility_rows}
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(max(12, len(labels) * 1.15), 7), layout="constrained")
    for row, label in zip(summary_rows, labels):
        samples = sorted(
            as_float(raw["frame_seconds"]) * 1_000.0
            for raw in raw_rows
            if combination_key(raw, parameter_fields) == combination_key(row, parameter_fields)
            and as_float(raw["frame_seconds"]) is not None
        )
        if samples:
            axis.step(samples, [(index + 1) / len(samples) for index in range(len(samples))], where="post", label=label)
    axis.set(title="Total frame latency ECDF", xlabel="milliseconds", ylabel="Cumulative proportion")
    axis.grid(alpha=0.3)
    axis.legend(title="Combination", fontsize=8, ncols=2)
    path = plots_dir / "frame_latency_ecdf.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    update_values = []
    update_labels = []
    for row, label in zip(summary_rows, labels):
        values = [
            as_float(raw["control_seconds"]) * 1_000.0
            for raw in raw_rows
            if combination_key(raw, parameter_fields) == combination_key(row, parameter_fields)
            and (as_float(raw["control_seconds"]) or 0.0) > 0.0
        ]
        if values:
            update_values.append(values)
            update_labels.append(label)
    if update_values:
        figure, axis = plt.subplots(figsize=(max(12, len(update_labels) * 1.15), 7), layout="constrained")
        axis.boxplot(update_values, tick_labels=update_labels, showfliers=True)
        axis.set(title="MPC optimization latency (update frames only)", ylabel="milliseconds", xlabel="parameter combination")
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.3)
        path = plots_dir / "mpc_update_latency_boxplot.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    if feasibility_rows:
        figure, axis = plt.subplots(figsize=(max(12, len(labels) * 1.15), 7), layout="constrained")
        points = []
        for row, label in zip(summary_rows, labels):
            feasibility = by_key[combination_key(row, parameter_fields)]
            point = axis.scatter(feasibility["frame_seconds_p95"] * 1_000.0, row["effective_fps_from_mean_frame"], s=80)
            axis.annotate(label.replace("\n", " "), point.get_offsets()[0], xytext=(5, 5), textcoords="offset points", fontsize=8)
            points.append(point)
        axis.set(title="Latency-throughput Pareto view", xlabel="p95 total frame latency (ms)", ylabel="effective FPS")
        axis.grid(alpha=0.3)
        path = plots_dir / "latency_fps_pareto.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    if reliability_rows:
        reliability_by_key = {combination_key(row, parameter_fields): row for row in reliability_rows}
        positions = list(range(len(labels)))
        success = [reliability_by_key.get(combination_key(row, parameter_fields), {}).get("optimizer_success_pct", 0.0) for row in summary_rows]
        recovery = [reliability_by_key.get(combination_key(row, parameter_fields), {}).get("control_recovery_pct", 0.0) for row in summary_rows]
        figure, axis = plt.subplots(figsize=(max(12, len(labels) * 1.15), 7), layout="constrained")
        axis.bar([position - 0.2 for position in positions], success, 0.4, label="optimizer success")
        axis.bar([position + 0.2 for position in positions], recovery, 0.4, label="fallback recovery")
        axis.set(title="MPC optimizer reliability", ylabel="MPC updates (%)", xlabel="parameter combination", ylim=(0, 100))
        axis.set_xticks(positions, labels)
        axis.tick_params(axis="x", labelsize=9)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        path = plots_dir / "mpc_optimizer_reliability.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    if {"control_interval_frames", "synthetic_animals"}.issubset(parameter_fields):
        intervals = sorted({str(row["control_interval_frames"]) for row in feasibility_rows}, key=float)
        animals = sorted({str(row["synthetic_animals"]) for row in feasibility_rows}, key=float)
        matrix = [
            [
                next(
                    (row["deadline_compliance_pct"] for row in feasibility_rows if str(row["control_interval_frames"]) == interval and str(row["synthetic_animals"]) == animal),
                    "",
                )
                for interval in intervals
            ]
            for animal in animals
        ]
        if all(value != "" for values in matrix for value in values):
            figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
            image = axis.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn")
            axis.set(title="Frames meeting the video deadline", xlabel="control interval (frames)", ylabel="synthetic animals")
            axis.set_xticks(range(len(intervals)), intervals)
            axis.set_yticks(range(len(animals)), animals)
            for row_index, values in enumerate(matrix):
                for column_index, value in enumerate(values):
                    axis.text(column_index, row_index, f"{value:.1f}%", ha="center", va="center")
            figure.colorbar(image, ax=axis, label="deadline compliance (%)")
            path = plots_dir / "deadline_compliance_heatmap.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            paths.append(path)
    return paths


def regenerate_plots(session_dir: Path) -> int:
    raw_path = session_dir / "frame_measurements.csv"
    summary_path = session_dir / "combination_summary.csv"
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("The session must contain frame_measurements.csv and combination_summary.csv.")
    with raw_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        raw_rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not raw_rows:
        raise ValueError("frame_measurements.csv contains no rows.")
    parameter_fields = [field for field in fieldnames if field not in {"video", "repeat", "frame", *METRICS}]
    with summary_path.open("r", newline="", encoding="utf-8") as source:
        summary_rows = list(csv.DictReader(source))
    for row in summary_rows:
        for column, value in row.items():
            if column not in parameter_fields and as_float(value) is not None:
                row[column] = float(value)
        for column in ("videos", "successful_measurement_runs", "measured_frames"):
            if column in row:
                row[column] = int(row[column])
    config_path = session_dir / "benchmark_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    videos = [project_path(path) for path in config.get("videos", [])]
    fps_by_video = read_video_fps(videos)
    feasibility_rows = build_feasibility_summary(raw_rows, parameter_fields, fps_by_video)
    reliability_rows = build_control_reliability_summary(session_dir, parameter_fields)
    per_video_rows = build_per_video_summary(raw_rows, parameter_fields, fps_by_video)
    enrich_combination_summary(summary_rows, feasibility_rows, reliability_rows, parameter_fields)
    summary_fields = [*parameter_fields, "videos", "successful_measurement_runs", "measured_frames"] + [f"{metric}_{name}" for metric in METRICS for name in ("mean", "median", "std", "min", "max")] + ["effective_fps_from_mean_frame", "frame_seconds_p95", "frame_seconds_p99", "deadline_frames", "deadline_compliance_pct", "mpc_updates_measured_frames", "control_update_seconds_mean", "control_update_seconds_median", "control_update_seconds_p95", "mpc_updates_all_frames", "optimizer_successes", "control_recoveries", "optimizer_success_pct", "control_recovery_pct"]
    write_csv(summary_path, summary_rows, summary_fields)
    per_video_fields = ["video", *parameter_fields, "measured_frames", "frame_seconds_mean", "frame_seconds_p95", "inference_seconds_mean", "control_update_seconds_median", "effective_fps", "deadline_compliance_pct"]
    write_csv(session_dir / "per_video_summary.csv", per_video_rows, per_video_fields)
    paths = create_plots(session_dir / "plots", raw_rows, summary_rows, parameter_fields)
    paths.extend(create_feasibility_plots(session_dir / "plots", raw_rows, summary_rows, feasibility_rows, reliability_rows, parameter_fields))
    paths.extend(create_by_video_plots(session_dir / "plots" / "by_video_comparison", per_video_rows, parameter_fields))
    create_global_plots_report(session_dir / "plots", summary_rows, parameter_fields)
    print(f"Regenerated {len(paths)} plots in: {session_dir / 'plots'}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.regenerate_plots is not None:
        return regenerate_plots(project_path(args.regenerate_plots))
    require_valid_args(args)
    videos_dir = project_path(args.videos_dir)
    model_path = project_path(args.model)
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    videos = selected_videos(videos_dir, args.videos)
    if not videos:
        raise FileNotFoundError(f"No direct video files found in: {videos_dir}")
    explicit_targets = parse_video_targets(videos_dir, videos, args.video_target)
    video_targets = {
        str(video): target_for_video(video, args.default_target, explicit_targets)
        for video in videos
    }

    grid = parse_parameter_grid(args)
    parameter_fields = [parameter_field(option) for option, _ in grid]
    fixed_options = {value.split("=", maxsplit=1)[0] for value in args.fixed_args if value.startswith("--")}
    grid_options = {option for option, _ in grid}
    duplicate_options = sorted(fixed_options & grid_options)
    if duplicate_options:
        raise ValueError(f"--fixed-args cannot override grid options: {', '.join(duplicate_options)}")
    combinations = [
        dict(zip(parameter_fields, values))
        for values in itertools.product(*(values for _, values in grid))
    ]
    session_name = args.session_name or f"mpc_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir = project_path(args.output_root) / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = session_dir / "logs"
    trials_dir = session_dir / "trials"
    logs_dir.mkdir()
    trials_dir.mkdir()
    (session_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "baseline_pipeline_args": BASELINE_PIPELINE_ARGS,
                "video_targets": video_targets,
                "videos": [str(video.relative_to(PROJECT_ROOT)) for video in videos],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_rows: list[dict[str, Any]] = []
    raw_frame_rows: list[dict[str, Any]] = []
    total_runs = len(videos) * len(combinations) * (args.warmups + args.measurements)
    sequence = 0
    randomizer = random.Random(args.seed)
    for phase, repetitions in (("warmup", args.warmups), ("measurement", args.measurements)):
        for repeat in range(1, repetitions + 1):
            for video in videos:
                ordered_combinations = combinations.copy()
                randomizer.shuffle(ordered_combinations)
                for combination in ordered_combinations:
                    sequence += 1
                    combination_name = "_".join(
                        f"{parameter_abbreviation(field)}-{value}" for field, value in combination.items()
                    ).replace("/", "_")
                    run_name = f"{video.stem}_{combination_name}"
                    if repetitions > 1:
                        run_name += f"_R-{repeat:02d}"
                    output_dir = trials_dir / phase / video.stem / combination_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    command = build_command(
                        args,
                        video,
                        output_dir,
                        run_name,
                        combination,
                        grid,
                        video_targets[str(video)],
                    )
                    log_path = logs_dir / f"{sequence:04d}_{run_name}.log"
                    print(f"[{sequence}/{total_runs}] {phase}: {video.name}, {combination_name}", flush=True)
                    started = time.perf_counter()
                    with log_path.open("w", encoding="utf-8") as log_file:
                        log_file.write(subprocess.list2cmdline(command) + "\n\n")
                        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
                    performance_path = output_dir / f"{run_name}_performance.csv"
                    row: dict[str, Any] = {
                        "sequence": sequence, "phase": phase, "repeat": repeat, "video": video.name,
                        **combination,
                        "return_code": completed.returncode, "wall_time_seconds": time.perf_counter() - started,
                        "performance_csv": str(performance_path.relative_to(session_dir)),
                        "log": str(log_path.relative_to(session_dir)), "frames_in_source": "", "error": "",
                    }
                    if completed.returncode == 0:
                        try:
                            row["frames_in_source"] = append_frame_rows(raw_frame_rows, performance_path, row, args.exclude_initial_frames, parameter_fields) if phase == "measurement" else ""
                        except (OSError, ValueError, KeyError) as error:
                            row["error"] = f"Could not read performance CSV: {error}"
                    else:
                        row["error"] = f"Pipeline exited with code {completed.returncode}."
                    run_rows.append(row)
                    write_csv(session_dir / "run_manifest.csv", run_rows, list(row))

    raw_fields = ["video", *parameter_fields, "repeat", "frame", *METRICS]
    write_csv(session_dir / "frame_measurements.csv", raw_frame_rows, raw_fields)
    per_frame_rows = aggregate_by_frame(raw_frame_rows, parameter_fields)
    per_frame_fields = ["video", *parameter_fields, "frame", "measurement_repeats"] + [f"{metric}_{name}" for metric in METRICS for name in ("mean", "median", "std", "min", "max")]
    write_csv(session_dir / "per_frame_statistics.csv", per_frame_rows, per_frame_fields)
    summary_rows = aggregate_combinations(raw_frame_rows, run_rows, parameter_fields)
    fps_by_video = read_video_fps(videos)
    feasibility_rows = build_feasibility_summary(raw_frame_rows, parameter_fields, fps_by_video)
    reliability_rows = build_control_reliability_summary(session_dir, parameter_fields)
    per_video_rows = build_per_video_summary(raw_frame_rows, parameter_fields, fps_by_video)
    enrich_combination_summary(summary_rows, feasibility_rows, reliability_rows, parameter_fields)
    summary_fields = [*parameter_fields, "videos", "successful_measurement_runs", "measured_frames"] + [f"{metric}_{name}" for metric in METRICS for name in ("mean", "median", "std", "min", "max")] + ["effective_fps_from_mean_frame", "frame_seconds_p95", "frame_seconds_p99", "deadline_frames", "deadline_compliance_pct", "mpc_updates_measured_frames", "control_update_seconds_mean", "control_update_seconds_median", "control_update_seconds_p95", "mpc_updates_all_frames", "optimizer_successes", "control_recoveries", "optimizer_success_pct", "control_recovery_pct"]
    write_csv(session_dir / "combination_summary.csv", summary_rows, summary_fields)
    per_video_fields = ["video", *parameter_fields, "measured_frames", "frame_seconds_mean", "frame_seconds_p95", "inference_seconds_mean", "control_update_seconds_median", "effective_fps", "deadline_compliance_pct"]
    write_csv(session_dir / "per_video_summary.csv", per_video_rows, per_video_fields)
    plots = []
    if raw_frame_rows:
        plots.extend(create_plots(session_dir / "plots", raw_frame_rows, summary_rows, parameter_fields))
        plots.extend(create_feasibility_plots(session_dir / "plots", raw_frame_rows, summary_rows, feasibility_rows, reliability_rows, parameter_fields))
        plots.extend(create_by_video_plots(session_dir / "plots" / "by_video_comparison", per_video_rows, parameter_fields))
        create_global_plots_report(session_dir / "plots", summary_rows, parameter_fields)
    failed = [row for row in run_rows if row["return_code"] != 0 or row["error"]]
    print(f"Results: {session_dir}", flush=True)
    print(f"Combination table: {session_dir / 'combination_summary.csv'}", flush=True)
    if plots:
        print(f"Plots: {session_dir / 'plots'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
