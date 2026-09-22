from __future__ import annotations

import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

from .cmff import CMFFGuidance
from .config import CMFFConfig, DynamicsConfig, MPCConfig
from .dynamics import SelfOrganizationRule
from .mpc import MPCController, RobotState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "dogRobot_v2_best.pt"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "mpc_herding"


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def box_center(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def extract_candidates(result, class_id: int) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    track_ids = None
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    candidates: list[dict] = []
    for index, raw_box in enumerate(boxes):
        if int(class_ids[index]) != class_id:
            continue
        x1, y1, x2, y2 = raw_box.tolist()
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        candidates.append(
            {
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "center": box_center((x1, y1, x2, y2)),
                "confidence": float(confidences[index]),
                "track_id": None if track_ids is None else int(track_ids[index]),
                "area": width * height,
            }
        )
    return candidates


def select_tracked_candidate(candidates: list[dict], active_track_id: int | None) -> dict | None:
    if not candidates:
        return None
    if active_track_id is not None:
        for candidate in candidates:
            if candidate["track_id"] == active_track_id:
                return candidate
    return max(candidates, key=lambda candidate: candidate["area"])


def build_ellipse_particles(
    box: tuple[float, float, float, float],
    count: int,
    previous_center: np.ndarray | None,
    previous_axes: np.ndarray | None,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = box_center(box)
    x1, y1, x2, y2 = box
    axes = np.array([max(1.0, (x2 - x1) * 0.5), max(1.0, (y2 - y1) * 0.5)], dtype=float)

    # Deterministic low-discrepancy-like sampling inside the observed flock ellipse.
    indices = np.arange(count, dtype=float)
    angles = indices * math.pi * (3.0 - math.sqrt(5.0))
    radii = np.sqrt((indices + 0.5) / count)
    local = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    positions = center[None, :] + local * axes[None, :]

    if previous_center is None:
        center_velocity = np.zeros(2, dtype=float)
        axis_velocity = np.zeros(2, dtype=float)
    else:
        center_velocity = (center - previous_center) / max(dt, 1e-6)
        axis_velocity = (axes - previous_axes) / max(dt, 1e-6) if previous_axes is not None else np.zeros(2, dtype=float)

    expansion_velocity = local * axis_velocity[None, :]
    velocities = center_velocity[None, :] + expansion_velocity
    return positions, velocities, center, axes


def target_from_args(args: argparse.Namespace, width: int, height: int) -> np.ndarray:
    if args.target is not None:
        x_text, y_text = args.target.split(",", maxsplit=1)
        return np.array([float(x_text), float(y_text)], dtype=float)

    if args.target_x is not None and args.target_y is not None:
        return np.array([float(args.target_x), float(args.target_y)], dtype=float)

    if args.target_x is not None or args.target_y is not None:
        raise ValueError("Use both --target-x and --target-y, or use --target x,y.")

    margin = float(args.target_margin)
    return np.array([width - margin, margin], dtype=float)


def initial_robot_state(
    first_flock_box: tuple[float, float, float, float],
    target: np.ndarray,
    width: int,
    height: int,
    offset_scale: float,
) -> RobotState:
    center = box_center(first_flock_box)
    rear = center - target
    rear_norm = np.linalg.norm(rear)
    if rear_norm < 1e-6:
        rear = np.array([0.0, 1.0], dtype=float)
    else:
        rear = rear / rear_norm
    x1, y1, x2, y2 = first_flock_box
    offset = max(x2 - x1, y2 - y1) * offset_scale
    position = center + rear * offset
    position[0] = float(np.clip(position[0], 0, width - 1))
    position[1] = float(np.clip(position[1], 0, height - 1))
    heading = math.atan2(center[1] - position[1], center[0] - position[0])
    return RobotState(float(position[0]), float(position[1]), heading)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_robot(frame: np.ndarray, state: RobotState, color: tuple[int, int, int]) -> None:
    center = (int(round(state.x)), int(round(state.y)))
    tip = (
        int(round(state.x + 22.0 * math.cos(state.theta))),
        int(round(state.y + 22.0 * math.sin(state.theta))),
    )
    cv2.circle(frame, center, 9, color, -1)
    cv2.arrowedLine(frame, center, tip, (255, 255, 255), 2, tipLength=0.35)


def smooth_point(
    previous: np.ndarray | None,
    current: np.ndarray,
    smoothing: float,
    max_step: float,
) -> np.ndarray:
    if previous is None:
        return current.copy()

    smoothing = float(np.clip(smoothing, 0.0, 0.99))
    filtered = smoothing * previous + (1.0 - smoothing) * current
    delta = filtered - previous
    distance = float(np.linalg.norm(delta))
    if max_step > 0.0 and distance > max_step:
        filtered = previous + delta / distance * max_step
    return filtered


def geometric_driving_point(
    center: np.ndarray,
    axes: np.ndarray,
    target: np.ndarray,
    frame_index: int,
    mode: str,
    driving_offset: float,
    sweep_amplitude: float,
    sweep_period: float,
    center_velocity: np.ndarray | None = None,
    lateral_drift_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    target_direction = target - center
    target_distance = float(np.linalg.norm(target_direction))
    if target_distance < 1e-6:
        rear_direction = np.array([0.0, 1.0], dtype=float)
    else:
        rear_direction = -target_direction / target_distance

    denominator = math.sqrt(
        (rear_direction[0] / max(1.0, axes[0])) ** 2
        + (rear_direction[1] / max(1.0, axes[1])) ** 2
    )
    edge_distance = 0.0 if denominator < 1e-6 else 1.0 / denominator
    driving_point = center + rear_direction * (edge_distance + driving_offset)

    lateral_direction = np.array([-rear_direction[1], rear_direction[0]], dtype=float)

    if mode == "sweep":
        period = max(1.0, sweep_period)
        phase = 2.0 * math.pi * frame_index / period
        driving_point = driving_point + lateral_direction * sweep_amplitude * math.sin(phase)

    if mode == "adaptive-sweep" and center_velocity is not None:
        lateral_velocity = float(np.dot(center_velocity, lateral_direction))
        lateral_offset = float(np.clip(lateral_drift_gain * lateral_velocity, -sweep_amplitude, sweep_amplitude))
        driving_point = driving_point + lateral_direction * lateral_offset

    return driving_point, rear_direction


def simulate_video_flock_only(
    video_path: Path,
    model_path: Path,
    output_video_path: Path,
    output_csv_path: Path,
    args: argparse.Namespace,
) -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is missing. Install it with: pip install opencv-python"
        )
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is missing. Install it with: pip install ultralytics"
        )

    video_path = resolve_project_path(video_path)
    model_path = resolve_project_path(model_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    dt = args.dt if args.dt is not None else 1.0 / fps

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_video_path}")

    print("=" * 70, flush=True)
    print("CMFF/MPC video mode using flock detections only", flush=True)
    print("=" * 70, flush=True)
    print(f"Video: {video_path}", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Frames: {total_frames}", flush=True)
    print(f"FPS: {fps:.3f}", flush=True)
    print(f"Detector: {'track + BoT-SORT' if args.use_tracker else 'predict'}", flush=True)
    print(f"Guidance mode: {args.guidance_mode}", flush=True)
    print(f"Output video: {output_video_path}", flush=True)
    print(f"Output CSV: {output_csv_path}", flush=True)
    print("Loading YOLO model...", flush=True)

    target = target_from_args(args, width, height)
    model = YOLO(str(model_path))
    print("YOLO model loaded.", flush=True)
    dynamics_config = DynamicsConfig(
        animal_sensing_radius=args.animal_sensing_radius,
        robot_sensing_radius=args.robot_sensing_radius,
        animal_max_speed=args.animal_max_speed,
        w_aggregation=args.w_aggregation,
        w_obstacle=0.0,
        w_robot=args.w_robot,
    )
    cmff_config = CMFFConfig(
        sigma=args.sigma,
        divergence_threshold=args.tau,
        eta=args.eta,
        driving_offset=args.driving_offset,
        grid_padding=args.grid_padding,
        grid_resolution=args.grid_resolution,
    )
    mpc_config = MPCConfig(
        horizon=args.horizon,
        dt=dt,
        min_speed=0.0,
        max_speed=args.robot_max_speed,
        max_omega=args.robot_max_omega,
        max_acceleration=args.robot_max_acceleration,
        max_angular_acceleration=args.robot_max_angular_acceleration,
        wheelbase=args.wheelbase,
        max_steering_angle=math.radians(args.max_steering_angle_deg),
        w_driving_point=args.w_driving_point,
        w_centroid_target=args.w_centroid_target,
        w_energy=args.w_energy,
        w_obstacle=0.0,
        max_iterations=args.max_iterations,
    )
    cmff = CMFFGuidance(cmff_config)
    controller = MPCController(mpc_config, dynamics_config, cmff, SelfOrganizationRule(args.rule))

    active_flock_track_id = None
    robot_state: RobotState | None = None
    previous_center: np.ndarray | None = None
    previous_axes: np.ndarray | None = None
    smoothed_driving_point: np.ndarray | None = None
    last_positions: np.ndarray | None = None
    last_velocities: np.ndarray | None = None
    robot_trajectory: deque[tuple[int, int]] = deque(maxlen=args.trail_length)
    rows: list[dict] = []
    frame_index = 0

    print()

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_start_time = time.perf_counter()
        if frame_index == 0 or frame_index % args.progress_interval == 0:
            print(f"Processing frame {frame_index + 1}/{total_frames}...", flush=True)

        detection_start_time = time.perf_counter()
        if args.debug_timing:
            print(f"  frame {frame_index}: starting YOLO detection", flush=True)
        if args.use_tracker:
            result = model.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0, 1] if args.draw_detected_dog else [0],
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.image_size,
                verbose=False,
            )[0]
        else:
            result = model.predict(
                source=frame,
                classes=[0, 1] if args.draw_detected_dog else [0],
                conf=args.confidence,
                iou=args.iou,
                imgsz=args.image_size,
                verbose=False,
            )[0]
        detection_seconds = time.perf_counter() - detection_start_time
        if args.debug_timing:
            print(f"  frame {frame_index}: detection finished in {detection_seconds:.2f}s", flush=True)

        flock_candidates = extract_candidates(result, class_id=0)
        selected_flock = select_tracked_candidate(flock_candidates, active_flock_track_id)
        detected_dogs = extract_candidates(result, class_id=1) if args.draw_detected_dog else []

        status = "NO_FLOCK"
        optimizer_success = False
        optimizer_cost = float("nan")
        guidance = None
        raw_driving_point = None
        rear_direction = None

        if selected_flock is not None:
            control_start_time = time.perf_counter()
            if args.debug_timing:
                print(f"  frame {frame_index}: starting guidance+MPC", flush=True)
            active_flock_track_id = selected_flock["track_id"]
            positions, velocities, previous_center, previous_axes = build_ellipse_particles(
                box=selected_flock["box"],
                count=args.synthetic_animals,
                previous_center=previous_center,
                previous_axes=previous_axes,
                dt=dt,
            )
            last_positions = positions
            last_velocities = velocities

            if robot_state is None:
                robot_state = initial_robot_state(
                    first_flock_box=selected_flock["box"],
                    target=target,
                    width=width,
                    height=height,
                    offset_scale=args.initial_robot_offset_scale,
            )

            if args.guidance_mode == "cmff":
                guidance = cmff.compute(positions, velocities, target)
                raw_driving_point = guidance.driving_point
            else:
                raw_driving_point, rear_direction = geometric_driving_point(
                    center=previous_center,
                    axes=previous_axes,
                    target=target,
                    frame_index=frame_index,
                    mode=args.guidance_mode,
                    driving_offset=args.driving_offset,
                    sweep_amplitude=args.sweep_amplitude,
                    sweep_period=args.sweep_period,
                    center_velocity=np.mean(velocities, axis=0) * dt,
                    lateral_drift_gain=args.lateral_drift_gain,
                )
            smoothed_driving_point = smooth_point(
                previous=smoothed_driving_point,
                current=raw_driving_point,
                smoothing=args.guidance_smoothing,
                max_step=args.guidance_max_step,
            )
            control, optimizer_success, optimizer_cost = controller.solve(
                robot_state=robot_state,
                herd_positions=positions,
                herd_velocities=velocities,
                target=target,
                obstacles=np.empty((0, 2), dtype=float),
                driving_point_override=smoothed_driving_point,
            )
            robot_state = RobotState.from_array(controller.step_robot_state(robot_state.as_array(), control))
            robot_state = RobotState(
                x=float(np.clip(robot_state.x, 0, width - 1)),
                y=float(np.clip(robot_state.y, 0, height - 1)),
                theta=robot_state.theta,
                v=robot_state.v,
                omega=robot_state.omega,
            )
            status = f"{args.guidance_mode.upper()}_MPC"
            control_seconds = time.perf_counter() - control_start_time
            if args.debug_timing:
                print(f"  frame {frame_index}: guidance+MPC finished in {control_seconds:.2f}s", flush=True)
        elif robot_state is not None and last_positions is not None and last_velocities is not None:
            control_start_time = time.perf_counter()
            if args.debug_timing:
                print(f"  frame {frame_index}: starting guidance+MPC from last flock", flush=True)
            if args.guidance_mode == "cmff":
                guidance = cmff.compute(last_positions, last_velocities, target)
                raw_driving_point = guidance.driving_point
            elif previous_center is not None and previous_axes is not None:
                raw_driving_point, rear_direction = geometric_driving_point(
                    center=previous_center,
                    axes=previous_axes,
                    target=target,
                    frame_index=frame_index,
                    mode=args.guidance_mode,
                    driving_offset=args.driving_offset,
                    sweep_amplitude=args.sweep_amplitude,
                    sweep_period=args.sweep_period,
                    center_velocity=np.mean(last_velocities, axis=0) * dt,
                    lateral_drift_gain=args.lateral_drift_gain,
                )
            smoothed_driving_point = smooth_point(
                previous=smoothed_driving_point,
                current=raw_driving_point,
                smoothing=args.guidance_smoothing,
                max_step=args.guidance_max_step,
            )
            control, optimizer_success, optimizer_cost = controller.solve(
                robot_state=robot_state,
                herd_positions=last_positions,
                herd_velocities=last_velocities,
                target=target,
                obstacles=np.empty((0, 2), dtype=float),
                driving_point_override=smoothed_driving_point,
            )
            robot_state = RobotState.from_array(controller.step_robot_state(robot_state.as_array(), control))
            status = f"LAST_FLOCK_{args.guidance_mode.upper()}"
            control_seconds = time.perf_counter() - control_start_time
            if args.debug_timing:
                print(f"  frame {frame_index}: guidance+MPC finished in {control_seconds:.2f}s", flush=True)
        else:
            control_seconds = 0.0

        annotated = frame.copy()
        cv2.circle(annotated, (int(round(target[0])), int(round(target[1]))), int(args.target_radius), (0, 220, 0), 2)
        draw_label(annotated, "Target", (int(target[0]) + 8, int(target[1]) - 8), (0, 220, 0))

        if selected_flock is not None:
            x1, y1, x2, y2 = selected_flock["box"]
            cv2.ellipse(
                annotated,
                (int(round(previous_center[0])), int(round(previous_center[1]))),
                (int(round(previous_axes[0])), int(round(previous_axes[1]))),
                0,
                0,
                360,
                (255, 170, 0),
                2,
            )
            center_point = (int(round(previous_center[0])), int(round(previous_center[1])))
            cv2.circle(annotated, center_point, 7, (0, 255, 255), -1)
            cv2.drawMarker(
                annotated,
                center_point,
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            draw_label(annotated, "Flock center", (center_point[0] + 10, center_point[1] - 10), (0, 255, 255))

        if guidance is not None and args.guidance_mode == "cmff":
            cv2.circle(annotated, tuple(np.round(guidance.q_div).astype(int)), 7, (0, 0, 255), -1)
            cv2.circle(annotated, tuple(np.round(raw_driving_point).astype(int)), 4, (180, 0, 180), 1)
            cv2.arrowedLine(
                annotated,
                tuple(np.round(guidance.q_div).astype(int)),
                tuple(np.round(guidance.q_div + 35.0 * guidance.fused_direction).astype(int)),
                (255, 0, 255),
                2,
                tipLength=0.3,
            )

        if smoothed_driving_point is not None:
            cv2.circle(annotated, tuple(np.round(smoothed_driving_point).astype(int)), 8, (255, 0, 255), -1)
            if previous_center is not None:
                cv2.line(
                    annotated,
                    tuple(np.round(previous_center).astype(int)),
                    tuple(np.round(smoothed_driving_point).astype(int)),
                    (255, 0, 255),
                    2,
                )

        for dog in detected_dogs:
            x1, y1, x2, y2 = dog["box"]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (80, 80, 255), 2)
            cv2.circle(annotated, tuple(np.round(dog["center"]).astype(int)), 4, (80, 80, 255), -1)
            draw_label(annotated, "detected dog (not used)", (int(x1), max(18, int(y1) - 6)), (80, 80, 255))

        if robot_state is not None:
            robot_trajectory.append((int(round(robot_state.x)), int(round(robot_state.y))))
            if len(robot_trajectory) >= 2:
                cv2.polylines(annotated, [np.array(robot_trajectory, dtype=np.int32)], False, (0, 0, 255), 2)
            draw_robot(annotated, robot_state, (0, 0, 255))
            draw_label(annotated, "simulated dog", (int(robot_state.x) + 12, int(robot_state.y) + 4), (0, 0, 255))

        draw_label(annotated, f"Frame {frame_index} | {status}", (15, 28), (255, 255, 255))
        draw_label(annotated, "Control uses flock detections only", (15, 55), (255, 255, 255))
        writer.write(annotated)

        rows.append(
            {
                "frame": frame_index,
                "status": status,
                "flock_detected": int(selected_flock is not None),
                "dog_detections": len(detected_dogs),
                "robot_x": "" if robot_state is None else f"{robot_state.x:.3f}",
                "robot_y": "" if robot_state is None else f"{robot_state.y:.3f}",
                "robot_theta": "" if robot_state is None else f"{robot_state.theta:.6f}",
                "robot_v": "" if robot_state is None else f"{robot_state.v:.3f}",
                "robot_omega": "" if robot_state is None else f"{robot_state.omega:.6f}",
                "target_x": f"{target[0]:.3f}",
                "target_y": f"{target[1]:.3f}",
                "flock_center_x": "" if previous_center is None else f"{previous_center[0]:.3f}",
                "flock_center_y": "" if previous_center is None else f"{previous_center[1]:.3f}",
                "guidance_mode": args.guidance_mode,
                "driving_point_x": "" if raw_driving_point is None else f"{raw_driving_point[0]:.3f}",
                "driving_point_y": "" if raw_driving_point is None else f"{raw_driving_point[1]:.3f}",
                "smoothed_driving_point_x": "" if smoothed_driving_point is None else f"{smoothed_driving_point[0]:.3f}",
                "smoothed_driving_point_y": "" if smoothed_driving_point is None else f"{smoothed_driving_point[1]:.3f}",
                "q_div_x": "" if guidance is None else f"{guidance.q_div[0]:.3f}",
                "q_div_y": "" if guidance is None else f"{guidance.q_div[1]:.3f}",
                "max_divergence": "" if guidance is None else f"{guidance.max_divergence:.6f}",
                "escape_detected": "" if guidance is None else int(guidance.escape_detected),
                "optimizer_success": int(optimizer_success),
                "optimizer_cost": "" if math.isnan(optimizer_cost) else f"{optimizer_cost:.6f}",
            }
        )

        frame_index += 1
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        frame_seconds = time.perf_counter() - frame_start_time
        if args.debug_timing or frame_index % args.progress_interval == 0:
            print(
                f"Processed {frame_index}/{total_frames} frames | "
                f"last={frame_seconds:.2f}s detection={detection_seconds:.2f}s "
                f"control={control_seconds:.2f}s status={status}",
                flush=True,
            )

    capture.release()
    writer.release()

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        if rows:
            writer_csv = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(rows)

    print("=" * 70, flush=True)
    print("Simulation finished", flush=True)
    print(f"Processed frames: {frame_index}", flush=True)
    print(f"Video: {output_video_path}", flush=True)
    print(f"CSV: {output_csv_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate dog motion from video using flock detections only.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--draw-detected-dog", action="store_true")
    parser.add_argument("--use-tracker", action="store_true", help="Use YOLO.track + BoT-SORT. Default uses faster YOLO.predict.")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--synthetic-animals", type=int, default=80)
    parser.add_argument("--rule", choices=[rule.value for rule in SelfOrganizationRule], default=SelfOrganizationRule.BOIDS.value)
    parser.add_argument("--target", type=str, default=None, help="Target point as x,y in image pixels. Example: --target 1850,350")
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-margin", type=float, default=80.0)
    parser.add_argument("--target-radius", type=float, default=45.0)
    parser.add_argument("--sigma", type=float, default=45.0)
    parser.add_argument("--tau", type=float, default=0.0001)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--guidance-mode", choices=["adaptive-sweep", "sweep", "rear", "cmff"], default="adaptive-sweep")
    parser.add_argument("--driving-offset", type=float, default=70.0)
    parser.add_argument("--sweep-amplitude", type=float, default=120.0)
    parser.add_argument("--sweep-period", type=float, default=90.0)
    parser.add_argument("--lateral-drift-gain", type=float, default=1.2)
    parser.add_argument("--guidance-smoothing", type=float, default=0.90, help="0 disables smoothing; values near 1 stabilize the pink point more.")
    parser.add_argument("--guidance-max-step", type=float, default=35.0, help="Maximum pink-point displacement per frame in pixels. 0 disables the limit.")
    parser.add_argument("--grid-padding", type=float, default=80.0)
    parser.add_argument("--grid-resolution", type=int, default=31)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--robot-max-speed", type=float, default=260.0)
    parser.add_argument("--robot-max-omega", type=float, default=4.0)
    parser.add_argument("--robot-max-acceleration", type=float, default=900.0)
    parser.add_argument("--robot-max-angular-acceleration", type=float, default=12.0)
    parser.add_argument("--wheelbase", type=float, default=80.0)
    parser.add_argument("--max-steering-angle-deg", type=float, default=35.0)
    parser.add_argument("--animal-sensing-radius", type=float, default=160.0)
    parser.add_argument("--robot-sensing-radius", type=float, default=300.0)
    parser.add_argument("--animal-max-speed", type=float, default=120.0)
    parser.add_argument("--w-aggregation", type=float, default=0.5)
    parser.add_argument("--w-robot", type=float, default=0.5)
    parser.add_argument("--w-driving-point", type=float, default=0.6)
    parser.add_argument("--w-centroid-target", type=float, default=0.3)
    parser.add_argument("--w-energy", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--initial-robot-offset-scale", type=float, default=1.4)
    parser.add_argument("--trail-length", type=int, default=240)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = resolve_project_path(args.video)
    run_name = args.run_name or f"{video_path.stem}_cmff_mpc_flock_only"
    output_dir = resolve_project_path(args.output_dir)
    simulate_video_flock_only(
        video_path=video_path,
        model_path=args.model,
        output_video_path=output_dir / f"{run_name}.mp4",
        output_csv_path=output_dir / f"{run_name}.csv",
        args=args,
    )


if __name__ == "__main__":
    main()
