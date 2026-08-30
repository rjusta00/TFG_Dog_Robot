import argparse
import csv
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch


SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flock_only_target_prediction.common import (
    DEFAULT_DETECTOR_MODEL_PATH,
    OUTPUT_ROOT,
    build_flock_feature_vector,
    build_linear_target_from_flock,
    clip_point,
    denormalize_target,
    estimate_target_from_flock_motion,
    extract_flock_candidates,
    load_detector,
    resolve_project_path,
    select_main_flock,
)
from flock_only_target_prediction.model import FlockTargetGRUPredictor
from simulate_robot_mpc import (
    build_reference_horizon,
    clamp,
    draw_predicted_trajectory,
    draw_reference_trajectory,
    draw_robot,
    smooth_guidance,
    solve_mpc,
    update_state,
)
from rear_dog_prediction.simulate_robot_mpc_trained_predictor import build_startup_control


def load_predictor(checkpoint_path: Path, device: torch.device):
    resolved_checkpoint_path = resolve_project_path(checkpoint_path)

    try:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)

    model = FlockTargetGRUPredictor(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
    std = np.array(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std, checkpoint


def select_safe_target(
    current_row: dict,
    predicted_point: tuple[float, float],
    linear_point: tuple[float, float] | None,
    last_target_point: tuple[int, int] | None,
) -> tuple[tuple[float, float], str]:
    flock_scale = max(
        1.0,
        float(current_row["flock_width"]),
        float(current_row["flock_height"]),
    )

    if last_target_point is not None:
        jump = math.hypot(
            predicted_point[0] - last_target_point[0],
            predicted_point[1] - last_target_point[1],
        )

        if jump > 0.35 * flock_scale:
            if linear_point is not None:
                return linear_point, "LINEAR_FALLBACK"

            return last_target_point, "TARGET_FROZEN"

    return predicted_point, "PREDICTED_FROM_FLOCK"


def smooth_target_point(
    raw_target_point: tuple[float, float],
    previous_target_point: tuple[float, float] | None,
    flock_scale: float,
) -> tuple[float, float]:
    if previous_target_point is None:
        return raw_target_point

    alpha = 0.22
    smoothed_x = alpha * raw_target_point[0] + (1.0 - alpha) * previous_target_point[0]
    smoothed_y = alpha * raw_target_point[1] + (1.0 - alpha) * previous_target_point[1]

    max_step = 0.12 * flock_scale
    delta_x = smoothed_x - previous_target_point[0]
    delta_y = smoothed_y - previous_target_point[1]
    delta_norm = math.hypot(delta_x, delta_y)

    if delta_norm > max_step and delta_norm > 1e-6:
        scale = max_step / delta_norm
        smoothed_x = previous_target_point[0] + delta_x * scale
        smoothed_y = previous_target_point[1] + delta_y * scale

    return smoothed_x, smoothed_y


def build_guidance_from_flock_predictor(
    video_path: Path,
    detector_model_path: Path,
    predictor_checkpoint_path: Path,
    confidence: float,
    image_size: int,
    predictor_device: str,
) -> tuple[dict[int, dict], float, int, int, int]:
    video_path = resolve_project_path(video_path)
    detector_model_path = resolve_project_path(detector_model_path)
    predictor_checkpoint_path = resolve_project_path(predictor_checkpoint_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not detector_model_path.exists():
        raise FileNotFoundError(f"Detector model not found: {detector_model_path}")

    if not predictor_checkpoint_path.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {predictor_checkpoint_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise RuntimeError("Invalid FPS in inference video.")

    device = torch.device(predictor_device)
    predictor, feature_mean, feature_std, checkpoint = load_predictor(predictor_checkpoint_path, device)
    history_length = int(checkpoint["history_length"])
    detector = load_detector(detector_model_path)

    print("=" * 70)
    print("BUILDING GUIDANCE WITH FLOCK-ONLY TARGET PREDICTOR")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Detector: {detector_model_path}")
    print(f"Predictor: {predictor_checkpoint_path}")
    print(f"Frames: {total_frames}")
    print()

    feature_history: deque[np.ndarray] = deque(maxlen=history_length)
    flock_history: deque[dict] = deque(maxlen=history_length)
    target_history: deque[tuple[int, int]] = deque(maxlen=history_length)
    active_flock_track_id = None
    previous_row = None
    last_target_point = None
    last_target_point_float = None
    guidance: dict[int, dict] = {}
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        result = detector.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=confidence,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )[0]

        flock_candidates = extract_flock_candidates(result)
        selected_flock = select_main_flock(flock_candidates, active_flock_track_id)

        if selected_flock is not None:
            active_flock_track_id = selected_flock["track_id"]
            current_row = {
                "frame_width": width,
                "frame_height": height,
                "flock_width": selected_flock["width"],
                "flock_height": selected_flock["height"],
                "flock_center_x": selected_flock["center"][0],
                "flock_center_y": selected_flock["center"][1],
            }

            if previous_row is not None and previous_row["frame_index"] != frame_index - 1:
                previous_row_for_feature = None
                feature_history.clear()
                flock_history.clear()
                target_history.clear()
            else:
                previous_row_for_feature = previous_row

            feature = build_flock_feature_vector(current_row, previous_row_for_feature)
            feature_history.append(feature)

            flock_snapshot = {
                **current_row,
                "frame_index": frame_index,
            }
            flock_history.append(flock_snapshot)

            linear_target = build_linear_target_from_flock(flock_history, target_history)
            heuristic_target = estimate_target_from_flock_motion(flock_history)

            if len(feature_history) >= history_length:
                feature_window = np.stack(feature_history, axis=0)[None, ...].astype(np.float32)
                normalized_feature_window = (feature_window - feature_mean) / feature_std

                with torch.no_grad():
                    prediction = predictor(torch.from_numpy(normalized_feature_window).to(device)).cpu().numpy()[0]

                predicted_point = denormalize_target(current_row, prediction)
                chosen_point, status = select_safe_target(current_row, predicted_point, linear_target, last_target_point)
                predicted_x, predicted_y = chosen_point

            elif linear_target is not None:
                predicted_x, predicted_y = linear_target
                status = "LINEAR_FALLBACK"

            elif heuristic_target is not None:
                predicted_x, predicted_y = heuristic_target
                status = "HEURISTIC_BACK_POSITION"

            elif last_target_point is not None:
                predicted_x, predicted_y = last_target_point
                status = "TARGET_FROZEN"

            else:
                predicted_x = float(selected_flock["center"][0])
                predicted_y = float(selected_flock["center"][1])
                status = "FLOCK_CENTER_FALLBACK"

            flock_scale = max(
                1.0,
                float(current_row["flock_width"]),
                float(current_row["flock_height"]),
            )

            predicted_x, predicted_y = smooth_target_point(
                raw_target_point=(predicted_x, predicted_y),
                previous_target_point=last_target_point_float,
                flock_scale=flock_scale,
            )

            target_point = clip_point((int(round(predicted_x)), int(round(predicted_y))), width, height)
            last_target_point = target_point
            last_target_point_float = (float(target_point[0]), float(target_point[1]))
            target_history.append(target_point)
            previous_row = {**current_row, "frame_index": frame_index}

            guidance[frame_index] = {
                "flock_x": float(selected_flock["center"][0]),
                "flock_y": float(selected_flock["center"][1]),
                "target_x": float(target_point[0]),
                "target_y": float(target_point[1]),
                "desired_x": float(target_point[0]),
                "desired_y": float(target_point[1]),
                "flock_box_x1": float(selected_flock["box"][0]),
                "flock_box_y1": float(selected_flock["box"][1]),
                "flock_box_x2": float(selected_flock["box"][2]),
                "flock_box_y2": float(selected_flock["box"][3]),
                "flock_confidence": float(selected_flock["confidence"]),
                "status": status,
            }

        elif last_target_point is not None:
            feature_history.clear()
            flock_history.clear()
            target_history.clear()
            previous_row = None

        frame_index += 1

        if frame_index % 150 == 0:
            print(f"Guidance progress: {frame_index}/{total_frames}")

    capture.release()

    if not guidance:
        raise RuntimeError("No guidance could be generated from the flock-only predictor.")

    return guidance, fps, width, height, total_frames


def draw_overlay(frame: np.ndarray, data: dict) -> None:
    x1 = int(round(data["flock_box_x1"]))
    y1 = int(round(data["flock_box_y1"]))
    x2 = int(round(data["flock_box_x2"]))
    y2 = int(round(data["flock_box_y2"]))
    flock_point = (int(data["flock_x"]), int(data["flock_y"]))
    target_point = (int(data["target_x"]), int(data["target_y"]))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
    cv2.circle(frame, flock_point, 10, (255, 255, 0), -1)
    cv2.putText(frame, "Flock center", (flock_point[0] + 15, flock_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, target_point, 18, (0, 255, 0), 4)
    cv2.putText(frame, "Predicted target", (target_point[0] + 24, target_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def simulate_robot_mpc_flock_target_predictor(
    video_path: Path,
    detector_model_path: Path,
    predictor_checkpoint_path: Path,
    robot_start_x: float,
    robot_start_y: float,
    initial_heading_degrees: float,
    max_speed: float,
    max_omega_degrees: float,
    max_acceleration: float,
    max_angular_acceleration_degrees: float,
    horizon: int,
    control_period: float,
    position_scale: float,
    stop_radius: float,
    weight_tracking: float,
    weight_energy: float,
    weight_terminal_position: float,
    weight_terminal_velocity: float,
    weight_smoothness: float,
    smoothing_alpha: float,
    confidence: float,
    image_size: int,
    predictor_device: str,
    run_name: str,
) -> None:
    guidance, fps, width, height, total_frames = build_guidance_from_flock_predictor(
        video_path,
        detector_model_path,
        predictor_checkpoint_path,
        confidence,
        image_size,
        predictor_device,
    )

    guidance = smooth_guidance(guidance, smoothing_alpha)
    capture = cv2.VideoCapture(str(resolve_project_path(video_path)))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    dt_video = 1.0 / fps
    control_interval_frames = max(1, int(round(control_period * fps)))
    dt_control = control_interval_frames / fps
    horizon_seconds = horizon * dt_control
    max_omega = math.radians(max_omega_degrees)
    max_angular_acceleration = math.radians(max_angular_acceleration_degrees)

    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_video = output_directory / f"{Path(video_path).stem}_flock_target_mpc.mp4"
    output_csv = output_directory / f"{Path(video_path).stem}_flock_target_mpc.csv"

    video_writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output video.")

    state = np.array([robot_start_x, robot_start_y, math.radians(initial_heading_degrees), 0.0, 0.0], dtype=float)
    current_control = np.array([0.0, 0.0], dtype=float)
    previous_solution = None
    robot_trajectory = deque(maxlen=500)
    last_guidance_data = None
    last_predicted_states = None
    last_reference_points = None
    last_mpc_cost = 0.0
    last_optimizer_success = True
    last_optimizer_iterations = 0
    next_control_frame = 0
    frame_index = 0
    startup_motion_frame = max(1, int(round(fps)))

    print("=" * 70)
    print("SIMULATING MPC WITH FLOCK-ONLY TARGET PREDICTOR")
    print("=" * 70)
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {total_frames}")
    print(f"Real MPC Ts: {dt_control:.4f} s")
    print(f"Horizon Hp: {horizon}")
    print(f"Time horizon: {horizon_seconds:.3f} s")
    print()

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "time_seconds", "robot_x", "robot_y", "theta_degrees", "velocity", "omega_degrees", "target_x", "target_y", "desired_x", "desired_y", "distance_to_desired", "status", "mpc_cost", "optimizer_success", "optimizer_iterations"])

        while True:
            success, frame = capture.read()
            if not success:
                break

            new_guidance_data = guidance.get(frame_index)
            if new_guidance_data is not None:
                last_guidance_data = new_guidance_data

            data = last_guidance_data
            mpc_updated = False

            if data is not None:
                if frame_index >= next_control_frame:
                    reference_points = build_reference_horizon(guidance, frame_index, data, horizon, control_interval_frames)
                    optimal_controls, predicted_states, mpc_cost, optimizer_success, optimizer_iterations = solve_mpc(
                        state=state,
                        reference_points=reference_points,
                        previous_control=current_control,
                        previous_solution=previous_solution,
                        horizon=horizon,
                        dt=dt_control,
                        position_scale=position_scale,
                        stop_radius=stop_radius,
                        max_speed=max_speed,
                        max_omega=max_omega,
                        max_acceleration=max_acceleration,
                        max_angular_acceleration=max_angular_acceleration,
                        weight_tracking=weight_tracking,
                        weight_energy=weight_energy,
                        weight_terminal_position=weight_terminal_position,
                        weight_terminal_velocity=weight_terminal_velocity,
                        weight_smoothness=weight_smoothness,
                    )

                    current_control = optimal_controls[0].copy()
                    distance_to_reference = math.hypot(float(data["desired_x"]) - float(state[0]), float(data["desired_y"]) - float(state[1]))

                    if frame_index >= startup_motion_frame and distance_to_reference > 120.0 and current_control[0] < 5.0:
                        current_control = build_startup_control(state, (float(data["desired_x"]), float(data["desired_y"])), dt_control, max_speed, max_omega, max_acceleration, max_angular_acceleration)

                    previous_solution = optimal_controls.copy()
                    last_predicted_states = predicted_states
                    last_reference_points = reference_points
                    last_mpc_cost = mpc_cost
                    last_optimizer_success = optimizer_success
                    last_optimizer_iterations = optimizer_iterations
                    mpc_updated = True
                    next_control_frame = frame_index + control_interval_frames

                state = update_state(state, current_control, dt_video)
                state[0] = clamp(state[0], 0.0, width - 1.0)
                state[1] = clamp(state[1], 0.0, height - 1.0)
                robot_point = (int(state[0]), int(state[1]))
                robot_trajectory.append(robot_point)
                distance_to_desired = math.hypot(float(data["desired_x"]) - float(state[0]), float(data["desired_y"]) - float(state[1]))

                draw_overlay(frame, data)
                draw_reference_trajectory(frame, last_reference_points)
                draw_predicted_trajectory(frame, last_predicted_states)

                if len(robot_trajectory) >= 2:
                    trajectory_points = np.array(robot_trajectory, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [trajectory_points], False, (255, 0, 255), 3)

                draw_robot(frame, robot_point, state[2])

                omega_degrees = math.degrees(current_control[1])
                theta_degrees = math.degrees(state[2])
                lines = [
                    f"MPC Hp: {horizon}",
                    f"MPC Ts: {dt_control:.3f} s",
                    f"Time horizon: {horizon_seconds:.2f} s",
                    f"v: {current_control[0]:.1f} px/s",
                    f"omega: {omega_degrees:.1f} deg/s",
                    f"Distance to ref: {distance_to_desired:.1f} px",
                    f"Target status: {data['status']}",
                    f"Optimizer: {'OK' if last_optimizer_success else 'FAILED'}",
                    f"MPC cost: {last_mpc_cost:.5f}",
                ]

                for line_index, text in enumerate(lines):
                    cv2.putText(frame, text, (30, 45 + line_index * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                if mpc_updated:
                    cv2.putText(frame, "MPC UPDATED", (30, 45 + len(lines) * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                csv_writer.writerow([frame_index, f"{frame_index / fps:.3f}", f"{state[0]:.3f}", f"{state[1]:.3f}", f"{theta_degrees:.3f}", f"{current_control[0]:.3f}", f"{omega_degrees:.3f}", f"{data['target_x']:.3f}", f"{data['target_y']:.3f}", f"{data['desired_x']:.3f}", f"{data['desired_y']:.3f}", f"{distance_to_desired:.3f}", data["status"], f"{last_mpc_cost:.8f}", last_optimizer_success, last_optimizer_iterations])

            video_writer.write(frame)
            frame_index += 1

            if frame_index % 200 == 0:
                print(f"Simulation progress: {frame_index}/{total_frames}")

    capture.release()
    video_writer.release()
    print()
    print("=" * 70)
    print("FLOCK-ONLY TARGET MPC FINISHED")
    print("=" * 70)
    print(f"Video: {output_video}")
    print(f"CSV: {output_csv}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate MPC using a flock-only target predictor.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_DETECTOR_MODEL_PATH)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--predictor-device", type=str, default="cpu")
    parser.add_argument("--robot-start-x", type=float, required=True)
    parser.add_argument("--robot-start-y", type=float, required=True)
    parser.add_argument("--initial-heading", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=250.0)
    parser.add_argument("--max-omega", type=float, default=120.0)
    parser.add_argument("--max-acceleration", type=float, default=150.0)
    parser.add_argument("--max-angular-acceleration", type=float, default=180.0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--control-period", type=float, default=0.1)
    parser.add_argument("--position-scale", type=float, default=200.0)
    parser.add_argument("--stop-radius", type=float, default=40.0)
    parser.add_argument("--weight-tracking", type=float, default=5.0)
    parser.add_argument("--weight-energy", type=float, default=0.01)
    parser.add_argument("--weight-terminal-position", type=float, default=10.0)
    parser.add_argument("--weight-terminal-velocity", type=float, default=5.0)
    parser.add_argument("--weight-smoothness", type=float, default=1.0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--name", type=str, default="flock_target_predictor_mpc")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    simulate_robot_mpc_flock_target_predictor(
        video_path=args.video,
        detector_model_path=args.detector_model,
        predictor_checkpoint_path=args.predictor_checkpoint,
        robot_start_x=args.robot_start_x,
        robot_start_y=args.robot_start_y,
        initial_heading_degrees=args.initial_heading,
        max_speed=args.max_speed,
        max_omega_degrees=args.max_omega,
        max_acceleration=args.max_acceleration,
        max_angular_acceleration_degrees=args.max_angular_acceleration,
        horizon=args.horizon,
        control_period=args.control_period,
        position_scale=args.position_scale,
        stop_radius=args.stop_radius,
        weight_tracking=args.weight_tracking,
        weight_energy=args.weight_energy,
        weight_terminal_position=args.weight_terminal_position,
        weight_terminal_velocity=args.weight_terminal_velocity,
        weight_smoothness=args.weight_smoothness,
        smoothing_alpha=args.smoothing_alpha,
        confidence=args.confidence,
        image_size=args.image_size,
        predictor_device=args.predictor_device,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
