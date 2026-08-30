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

from simulate_robot_mpc import (
    build_reference_horizon,
    clamp,
    draw_predicted_trajectory,
    draw_reference_trajectory,
    draw_robot,
    resolve_project_path,
    smooth_guidance,
    solve_mpc,
    update_state,
)
from rear_dog_prediction.common import (
    DEFAULT_MODEL_PATH,
    OUTPUT_ROOT,
    RearDogSelector,
    build_feature_vector,
    calculate_motion_vector,
    clip_point,
    denormalize_prediction,
    extract_tracked_candidates,
    load_yolo_model,
    select_main_flock,
)
from rear_dog_prediction.model import RearDogGRUPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def select_startup_dog_candidate(
    dog_candidates: list[dict],
    flock_center: tuple[int, int],
    preferred_track_id: int | None,
    last_target_point: tuple[int, int] | None,
) -> dict | None:
    if not dog_candidates:
        return None

    if preferred_track_id is not None:
        for candidate in dog_candidates:
            if candidate["track_id"] == preferred_track_id:
                return candidate

    if last_target_point is not None:
        return min(
            dog_candidates,
            key=lambda candidate: math.hypot(
                candidate["center"][0] - last_target_point[0],
                candidate["center"][1] - last_target_point[1],
            ),
        )

    return max(
        dog_candidates,
        key=lambda candidate: candidate["confidence"],
    )


def build_linear_prediction(
    observations: deque[dict],
    future_seconds: float,
) -> tuple[float, float] | None:
    if len(observations) < 2:
        return None

    times = np.array(
        [entry["time_seconds"] for entry in observations],
        dtype=float,
    )
    xs = np.array(
        [entry["dog_center_x"] for entry in observations],
        dtype=float,
    )
    ys = np.array(
        [entry["dog_center_y"] for entry in observations],
        dtype=float,
    )

    reference_time = float(times[-1])
    relative_times = times - reference_time

    if np.allclose(relative_times, relative_times[0]):
        return float(xs[-1]), float(ys[-1])

    x_slope, x_intercept = np.polyfit(
        relative_times,
        xs,
        deg=1,
    )
    y_slope, y_intercept = np.polyfit(
        relative_times,
        ys,
        deg=1,
    )

    return (
        float(x_slope * future_seconds + x_intercept),
        float(y_slope * future_seconds + y_intercept),
    )


def select_safe_target_point(
    current_row: dict,
    observed_point: tuple[float, float],
    gru_prediction: tuple[float, float] | None,
    linear_prediction: tuple[float, float] | None,
    last_target_point: tuple[int, int] | None,
) -> tuple[tuple[float, float], str]:
    flock_width = max(
        1.0,
        float(current_row["flock_width"]),
    )

    flock_height = max(
        1.0,
        float(current_row["flock_height"]),
    )

    flock_scale = max(
        flock_width,
        flock_height,
    )

    maximum_jump_from_observed = 0.45 * flock_scale
    maximum_disagreement = 0.35 * flock_scale
    maximum_step_from_previous_target = 0.30 * flock_scale

    def is_reasonable_candidate(
        candidate_point: tuple[float, float] | None,
    ) -> bool:
        if candidate_point is None:
            return False

        candidate_x, candidate_y = candidate_point

        if math.hypot(
            candidate_x - observed_point[0],
            candidate_y - observed_point[1],
        ) > maximum_jump_from_observed:
            return False

        if last_target_point is not None:
            if math.hypot(
                candidate_x - last_target_point[0],
                candidate_y - last_target_point[1],
            ) > maximum_step_from_previous_target:
                return False

        return True

    safe_linear_prediction = (
        linear_prediction
        if is_reasonable_candidate(linear_prediction)
        else None
    )

    if gru_prediction is None:
        if safe_linear_prediction is not None:
            return safe_linear_prediction, "LINEAR_FALLBACK"

        return observed_point, "OBSERVED_DOG"

    gru_to_observed = math.hypot(
        gru_prediction[0] - observed_point[0],
        gru_prediction[1] - observed_point[1],
    )

    if gru_to_observed > maximum_jump_from_observed:
        if safe_linear_prediction is not None:
            return safe_linear_prediction, "LINEAR_FALLBACK"

        return observed_point, "OBSERVED_DOG"

    if last_target_point is not None:
        gru_to_previous_target = math.hypot(
            gru_prediction[0] - last_target_point[0],
            gru_prediction[1] - last_target_point[1],
        )

        if gru_to_previous_target > maximum_step_from_previous_target:
            if safe_linear_prediction is not None:
                return safe_linear_prediction, "LINEAR_FALLBACK"

            return observed_point, "OBSERVED_DOG"

    if safe_linear_prediction is not None:
        gru_to_linear = math.hypot(
            gru_prediction[0] - safe_linear_prediction[0],
            gru_prediction[1] - safe_linear_prediction[1],
        )

        if gru_to_linear > maximum_disagreement:
            return safe_linear_prediction, "LINEAR_FALLBACK"

    return gru_prediction, "PREDICTED"


def build_startup_control(
    state: np.ndarray,
    desired_point: tuple[float, float],
    dt_control: float,
    max_speed: float,
    max_omega: float,
    max_acceleration: float,
    max_angular_acceleration: float,
) -> np.ndarray:
    dx = desired_point[0] - float(state[0])
    dy = desired_point[1] - float(state[1])
    desired_heading = math.atan2(dy, dx)
    heading_error = desired_heading - float(state[2])

    while heading_error > math.pi:
        heading_error -= 2.0 * math.pi

    while heading_error < -math.pi:
        heading_error += 2.0 * math.pi

    target_speed = max_speed * max(0.35, math.cos(heading_error))
    target_omega = max(
        -max_omega,
        min(max_omega, 2.0 * heading_error),
    )

    maximum_delta_v = max_acceleration * dt_control
    maximum_delta_omega = max_angular_acceleration * dt_control

    startup_speed = min(target_speed, maximum_delta_v)
    startup_omega = max(
        -maximum_delta_omega,
        min(maximum_delta_omega, target_omega),
    )

    return np.array(
        [startup_speed, startup_omega],
        dtype=float,
    )


def load_predictor(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RearDogGRUPredictor, np.ndarray, np.ndarray, dict]:
    resolved_checkpoint_path = resolve_project_path(
        checkpoint_path
    )

    try:
        checkpoint = torch.load(
            resolved_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            resolved_checkpoint_path,
            map_location=device,
        )

    model = RearDogGRUPredictor(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    mean = np.array(
        checkpoint["feature_mean"],
        dtype=np.float32,
    )

    std = np.array(
        checkpoint["feature_std"],
        dtype=np.float32,
    )

    return model, mean, std, checkpoint


def draw_target_marker(
    frame: np.ndarray,
    target_point: tuple[int, int],
) -> None:
    cv2.circle(
        frame,
        target_point,
        18,
        (0, 255, 0),
        4,
    )

    cv2.putText(
        frame,
        "Predicted target",
        (target_point[0] + 24, target_point[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def build_guidance_from_trained_predictor(
    video_path: Path,
    detector_model_path: Path,
    predictor_checkpoint_path: Path,
    confidence: float,
    image_size: int,
    motion_window: int,
    motion_dead_zone: float,
    rear_projection_threshold: float,
    lateral_penalty: float,
    device_name: str,
) -> tuple[dict[int, dict], float, int, int, int]:
    video_path = resolve_project_path(
        video_path
    )
    detector_model_path = resolve_project_path(
        detector_model_path
    )
    predictor_checkpoint_path = resolve_project_path(
        predictor_checkpoint_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not detector_model_path.exists():
        raise FileNotFoundError(
            f"Detector model not found: {detector_model_path}"
        )

    if not predictor_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Predictor checkpoint not found: {predictor_checkpoint_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "Invalid FPS in inference video."
        )

    device = torch.device(device_name)
    predictor, feature_mean, feature_std, checkpoint = load_predictor(
        checkpoint_path=predictor_checkpoint_path,
        device=device,
    )

    history_length = int(
        checkpoint["history_length"]
    )

    detector = load_yolo_model(
        detector_model_path
    )

    flock_trajectory: deque[tuple[int, int]] = deque(
        maxlen=max(2 * motion_window, 60),
    )
    feature_history: deque[np.ndarray] = deque(
        maxlen=history_length,
    )
    observation_history: deque[dict] = deque(
        maxlen=history_length,
    )
    active_flock_track_id = None
    last_rear_direction = None
    last_target_point = None
    previous_visible_row = None
    selector = RearDogSelector(
        min_projection=rear_projection_threshold,
        lateral_penalty=lateral_penalty,
    )

    guidance: dict[int, dict] = {}
    frame_index = 0

    print("=" * 70)
    print("BUILDING GUIDANCE WITH TRAINED REAR DOG PREDICTOR")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Predictor: {predictor_checkpoint_path}")
    print()

    while True:
        success, frame = capture.read()

        if not success:
            break

        result = detector.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0, 1],
            conf=confidence,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )[0]

        flock_candidates, dog_candidates = extract_tracked_candidates(
            result
        )

        selected_flock = select_main_flock(
            candidates=flock_candidates,
            active_track_id=active_flock_track_id,
        )

        if selected_flock is not None:
            active_flock_track_id = selected_flock["track_id"]
            flock_center = selected_flock["center"]
            flock_trajectory.append(
                flock_center
            )

            motion_direction, _ = calculate_motion_vector(
                trajectory=flock_trajectory,
                window=motion_window,
                dead_zone=motion_dead_zone,
            )

            if motion_direction is not None:
                last_rear_direction = -motion_direction

            selected_dog = selector.select(
                dog_candidates=dog_candidates,
                flock_center=flock_center,
                rear_direction=last_rear_direction,
            )

            if selected_dog is None:
                preferred_track_id = selector.active_track_id
                selected_dog = select_startup_dog_candidate(
                    dog_candidates=dog_candidates,
                    flock_center=flock_center,
                    preferred_track_id=preferred_track_id,
                    last_target_point=last_target_point,
                )

                if selected_dog is not None:
                    selector.active_track_id = selected_dog["track_id"]

            if selected_dog is not None:
                current_row = {
                    "frame_width": width,
                    "frame_height": height,
                    "flock_width": selected_flock["width"],
                    "flock_height": selected_flock["height"],
                    "dog_width": selected_dog["width"],
                    "dog_height": selected_dog["height"],
                    "flock_center_x": selected_flock["center"][0],
                    "flock_center_y": selected_flock["center"][1],
                    "dog_center_x": selected_dog["center"][0],
                    "dog_center_y": selected_dog["center"][1],
                    "dog_track_id": (
                        ""
                        if selected_dog["track_id"] is None
                        else selected_dog["track_id"]
                    ),
                }

                if previous_visible_row is not None:
                    frame_gap = frame_index - int(
                        previous_visible_row["frame_index"]
                    )
                    same_track = (
                        current_row["dog_track_id"]
                        == previous_visible_row["dog_track_id"]
                    )

                    if frame_gap != 1 or not same_track:
                        previous_row_for_feature = None
                        feature_history.clear()
                        observation_history.clear()
                    else:
                        previous_row_for_feature = previous_visible_row
                else:
                    previous_row_for_feature = None

                feature = build_feature_vector(
                    current_row=current_row,
                    previous_row=previous_row_for_feature,
                )

                feature_history.append(feature)
                observation_history.append(
                    {
                        "time_seconds": frame_index / fps,
                        "dog_center_x": float(selected_dog["center"][0]),
                        "dog_center_y": float(selected_dog["center"][1]),
                    }
                )

                observed_point = (
                    float(selected_dog["center"][0]),
                    float(selected_dog["center"][1]),
                )

                linear_prediction = build_linear_prediction(
                    observations=observation_history,
                    future_seconds=float(checkpoint["prediction_horizon_seconds"]),
                )

                gru_prediction = None

                if len(feature_history) >= history_length:
                    feature_window = np.stack(
                        feature_history,
                        axis=0,
                    )[None, ...].astype(np.float32)

                    normalized_feature_window = (
                        feature_window - feature_mean
                    ) / feature_std

                    with torch.no_grad():
                        prediction = predictor(
                            torch.from_numpy(normalized_feature_window).to(device)
                        ).cpu().numpy()[0]

                    predicted_x, predicted_y = denormalize_prediction(
                        current_row=current_row,
                        prediction=prediction,
                    )

                    gru_prediction = (
                        predicted_x,
                        predicted_y,
                    )

                else:
                    if last_rear_direction is None:
                        target_status = "STARTUP_DOG"
                    else:
                        target_status = "OBSERVED_DOG"

                if len(feature_history) >= history_length:
                    chosen_point, target_status = select_safe_target_point(
                        current_row=current_row,
                        observed_point=observed_point,
                        gru_prediction=gru_prediction,
                        linear_prediction=linear_prediction,
                        last_target_point=last_target_point,
                    )
                    predicted_x, predicted_y = chosen_point
                elif linear_prediction is not None and last_rear_direction is not None:
                    chosen_point, target_status = select_safe_target_point(
                        current_row=current_row,
                        observed_point=observed_point,
                        gru_prediction=None,
                        linear_prediction=linear_prediction,
                        last_target_point=last_target_point,
                    )
                    predicted_x, predicted_y = chosen_point
                else:
                    predicted_x, predicted_y = observed_point

                target_point = clip_point(
                    point=(
                        int(round(predicted_x)),
                        int(round(predicted_y)),
                    ),
                    frame_width=width,
                    frame_height=height,
                )

                last_target_point = target_point
                desired_point = target_point
                previous_visible_row = {
                    **current_row,
                    "frame_index": frame_index,
                }

                guidance[frame_index] = {
                    "flock_x": float(flock_center[0]),
                    "flock_y": float(flock_center[1]),
                    "target_x": float(target_point[0]),
                    "target_y": float(target_point[1]),
                    "desired_x": float(desired_point[0]),
                    "desired_y": float(desired_point[1]),
                    "flock_box_x1": float(selected_flock["box"][0]),
                    "flock_box_y1": float(selected_flock["box"][1]),
                    "flock_box_x2": float(selected_flock["box"][2]),
                    "flock_box_y2": float(selected_flock["box"][3]),
                    "dog_box_x1": float(selected_dog["box"][0]),
                    "dog_box_y1": float(selected_dog["box"][1]),
                    "dog_box_x2": float(selected_dog["box"][2]),
                    "dog_box_y2": float(selected_dog["box"][3]),
                    "dog_center_x": float(selected_dog["center"][0]),
                    "dog_center_y": float(selected_dog["center"][1]),
                    "dog_track_id": (
                        ""
                        if selected_dog["track_id"] is None
                        else int(selected_dog["track_id"])
                    ),
                    "dog_confidence": float(selected_dog["confidence"]),
                    "dogs_in_frame": int(len(dog_candidates)),
                    "status": target_status,
                }

            elif last_target_point is not None:
                selector.clear_active_track()
                feature_history.clear()
                observation_history.clear()
                previous_visible_row = None
                guidance[frame_index] = {
                    "flock_x": float(flock_center[0]),
                    "flock_y": float(flock_center[1]),
                    "target_x": float(last_target_point[0]),
                    "target_y": float(last_target_point[1]),
                    "desired_x": float(last_target_point[0]),
                    "desired_y": float(last_target_point[1]),
                    "flock_box_x1": float(selected_flock["box"][0]),
                    "flock_box_y1": float(selected_flock["box"][1]),
                    "flock_box_x2": float(selected_flock["box"][2]),
                    "flock_box_y2": float(selected_flock["box"][3]),
                    "dog_box_x1": "",
                    "dog_box_y1": "",
                    "dog_box_x2": "",
                    "dog_box_y2": "",
                    "dog_center_x": "",
                    "dog_center_y": "",
                    "dog_track_id": "",
                    "dog_confidence": 0.0,
                    "dogs_in_frame": int(len(dog_candidates)),
                    "status": "TARGET_FROZEN",
                }

        frame_index += 1

        if frame_index % 150 == 0:
            print(
                f"Processed {frame_index}/{total_frames} frames"
            )

    capture.release()

    if not guidance:
        raise RuntimeError(
            "No guidance could be generated with the trained predictor."
        )

    return guidance, fps, width, height, total_frames


def draw_overlay(
    frame: np.ndarray,
    data: dict,
) -> None:
    flock_box = (
        data["flock_box_x1"],
        data["flock_box_y1"],
        data["flock_box_x2"],
        data["flock_box_y2"],
    )

    cv2.rectangle(
        frame,
        (int(round(flock_box[0])), int(round(flock_box[1]))),
        (int(round(flock_box[2])), int(round(flock_box[3]))),
        (255, 255, 0),
        3,
    )

    flock_point = (
        int(data["flock_x"]),
        int(data["flock_y"]),
    )

    target_point = (
        int(data["target_x"]),
        int(data["target_y"]),
    )

    cv2.circle(
        frame,
        flock_point,
        10,
        (255, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        "Flock center",
        (flock_point[0] + 15, flock_point[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if data["dog_box_x1"] != "":
        dog_box = (
            int(round(float(data["dog_box_x1"]))),
            int(round(float(data["dog_box_y1"]))),
            int(round(float(data["dog_box_x2"]))),
            int(round(float(data["dog_box_y2"]))),
        )

        cv2.rectangle(
            frame,
            (dog_box[0], dog_box[1]),
            (dog_box[2], dog_box[3]),
            (0, 200, 0),
            3,
        )

        cv2.putText(
            frame,
            f"Rear dog track {data['dog_track_id']}",
            (dog_box[0], max(30, dog_box[1] - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    draw_target_marker(
        frame=frame,
        target_point=target_point,
    )


def simulate_robot_mpc_trained_predictor(
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
    motion_window: int,
    motion_dead_zone: float,
    rear_projection_threshold: float,
    lateral_penalty: float,
    predictor_device: str,
    run_name: str,
) -> None:
    guidance, fps, width, height, total_frames = build_guidance_from_trained_predictor(
        video_path=video_path,
        detector_model_path=detector_model_path,
        predictor_checkpoint_path=predictor_checkpoint_path,
        confidence=confidence,
        image_size=image_size,
        motion_window=motion_window,
        motion_dead_zone=motion_dead_zone,
        rear_projection_threshold=rear_projection_threshold,
        lateral_penalty=lateral_penalty,
        device_name=predictor_device,
    )

    guidance = smooth_guidance(
        guidance=guidance,
        alpha=smoothing_alpha,
    )

    capture = cv2.VideoCapture(
        str(resolve_project_path(video_path))
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    dt_video = 1.0 / fps
    control_interval_frames = max(1, int(round(control_period * fps)))
    dt_control = control_interval_frames / fps
    horizon_seconds = horizon * dt_control
    max_omega = math.radians(max_omega_degrees)
    max_angular_acceleration = math.radians(max_angular_acceleration_degrees)

    output_directory = OUTPUT_ROOT / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_video = output_directory / f"{Path(video_path).stem}_trained_predictor_mpc.mp4"
    output_csv = output_directory / f"{Path(video_path).stem}_trained_predictor_mpc.csv"

    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Could not create the output video."
        )

    state = np.array(
        [
            robot_start_x,
            robot_start_y,
            math.radians(initial_heading_degrees),
            0.0,
            0.0,
        ],
        dtype=float,
    )

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

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame",
                "time_seconds",
                "robot_x",
                "robot_y",
                "theta_degrees",
                "velocity",
                "omega_degrees",
                "target_x",
                "target_y",
                "desired_x",
                "desired_y",
                "distance_to_desired",
                "status",
                "dog_track_id",
                "mpc_cost",
                "optimizer_success",
                "optimizer_iterations",
            ]
        )

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
                    reference_points = build_reference_horizon(
                        guidance=guidance,
                        frame_index=frame_index,
                        current_data=data,
                        horizon=horizon,
                        control_interval_frames=control_interval_frames,
                    )

                    (
                        optimal_controls,
                        predicted_states,
                        mpc_cost,
                        optimizer_success,
                        optimizer_iterations,
                    ) = solve_mpc(
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

                    distance_to_reference = math.hypot(
                        float(data["desired_x"]) - float(state[0]),
                        float(data["desired_y"]) - float(state[1]),
                    )

                    if (
                        frame_index >= startup_motion_frame
                        and distance_to_reference > 120.0
                        and current_control[0] < 5.0
                    ):
                        current_control = build_startup_control(
                            state=state,
                            desired_point=(
                                float(data["desired_x"]),
                                float(data["desired_y"]),
                            ),
                            dt_control=dt_control,
                            max_speed=max_speed,
                            max_omega=max_omega,
                            max_acceleration=max_acceleration,
                            max_angular_acceleration=max_angular_acceleration,
                        )

                    previous_solution = optimal_controls.copy()
                    last_predicted_states = predicted_states
                    last_reference_points = reference_points
                    last_mpc_cost = mpc_cost
                    last_optimizer_success = optimizer_success
                    last_optimizer_iterations = optimizer_iterations
                    mpc_updated = True
                    next_control_frame = frame_index + control_interval_frames

                state = update_state(
                    state=state,
                    control=current_control,
                    dt=dt_video,
                )

                state[0] = clamp(state[0], 0.0, width - 1.0)
                state[1] = clamp(state[1], 0.0, height - 1.0)

                robot_point = (int(state[0]), int(state[1]))
                robot_trajectory.append(robot_point)
                distance_to_desired = math.hypot(
                    data["desired_x"] - state[0],
                    data["desired_y"] - state[1],
                )

                draw_overlay(frame=frame, data=data)
                draw_reference_trajectory(frame=frame, reference_points=last_reference_points)
                draw_predicted_trajectory(frame=frame, predicted_states=last_predicted_states)

                if len(robot_trajectory) >= 2:
                    trajectory_points = np.array(robot_trajectory, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [trajectory_points], False, (255, 0, 255), 3)

                draw_robot(frame=frame, robot_position=robot_point, theta=state[2])

                omega_degrees = math.degrees(current_control[1])
                theta_degrees = math.degrees(state[2])
                optimizer_text = "OK" if last_optimizer_success else "FAILED"

                lines = [
                    f"MPC Hp: {horizon}",
                    f"MPC Ts: {dt_control:.3f} s",
                    f"Time horizon: {horizon_seconds:.2f} s",
                    f"v: {current_control[0]:.1f} px/s",
                    f"omega: {omega_degrees:.1f} deg/s",
                    f"Distance to ref: {distance_to_desired:.1f} px",
                    f"Target status: {data['status']}",
                    f"Dogs in frame: {int(data['dogs_in_frame'])}",
                    f"Optimizer: {optimizer_text}",
                    f"MPC cost: {last_mpc_cost:.5f}",
                ]

                for line_index, text in enumerate(lines):
                    cv2.putText(
                        frame,
                        text,
                        (30, 45 + line_index * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if mpc_updated:
                    cv2.putText(
                        frame,
                        "MPC UPDATED",
                        (30, 45 + len(lines) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{frame_index / fps:.3f}",
                        f"{state[0]:.3f}",
                        f"{state[1]:.3f}",
                        f"{theta_degrees:.3f}",
                        f"{current_control[0]:.3f}",
                        f"{omega_degrees:.3f}",
                        f"{data['target_x']:.3f}",
                        f"{data['target_y']:.3f}",
                        f"{data['desired_x']:.3f}",
                        f"{data['desired_y']:.3f}",
                        f"{distance_to_desired:.3f}",
                        data["status"],
                        data["dog_track_id"],
                        f"{last_mpc_cost:.8f}",
                        last_optimizer_success,
                        last_optimizer_iterations,
                    ]
                )

            video_writer.write(frame)
            frame_index += 1

            if frame_index % 200 == 0:
                print(f"{frame_index}/{total_frames}")

    capture.release()
    video_writer.release()

    print()
    print("=" * 70)
    print("TRAINED PREDICTOR MPC FINISHED")
    print("=" * 70)
    print(f"Video: {output_video}")
    print(f"CSV: {output_csv}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate robot MPC using a trained rear dog predictor on a new video."
        )
    )

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--motion-window", type=int, default=12)
    parser.add_argument("--motion-dead-zone", type=float, default=10.0)
    parser.add_argument("--rear-projection-threshold", type=float, default=15.0)
    parser.add_argument("--lateral-penalty", type=float, default=0.35)
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
    parser.add_argument("--name", type=str, default="rear_dog_trained_predictor_mpc")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    simulate_robot_mpc_trained_predictor(
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
        motion_window=args.motion_window,
        motion_dead_zone=args.motion_dead_zone,
        rear_projection_threshold=args.rear_projection_threshold,
        lateral_penalty=args.lateral_penalty,
        predictor_device=args.predictor_device,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
