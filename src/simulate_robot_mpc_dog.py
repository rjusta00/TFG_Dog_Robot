import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dogRobot_v2_best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "robot_mpc_dog"
)


def calculate_box_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int]:
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))

    return center_x, center_y


class DogTargetSelector:
    """
    Mantiene un perro objetivo estable cuando aparecen
    uno o varios perros a la vez.
    """

    def __init__(self) -> None:
        self.active_track_id: int | None = None
        self.last_center: tuple[int, int] | None = None

    def select(
        self,
        candidates: list[dict],
        fallback_point: tuple[float, float],
    ) -> dict | None:
        if not candidates:
            return None

        if self.active_track_id is not None:
            for candidate in candidates:
                if candidate["track_id"] == self.active_track_id:
                    self.last_center = candidate["center"]
                    return candidate

        if self.last_center is not None:
            reference_x, reference_y = self.last_center
        else:
            reference_x, reference_y = fallback_point

        selected = min(
            candidates,
            key=lambda candidate: math.hypot(
                candidate["center"][0] - reference_x,
                candidate["center"][1] - reference_y,
            ),
        )

        self.active_track_id = selected["track_id"]
        self.last_center = selected["center"]

        return selected


def extract_dog_candidates(result) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    track_ids = None

    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)

    candidates = []

    for index, box in enumerate(boxes_xyxy):
        if class_ids[index] != 1:
            continue

        x1, y1, x2, y2 = box.tolist()
        center = calculate_box_center(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )

        track_id = None

        if track_ids is not None:
            track_id = int(track_ids[index])

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "center": center,
                "confidence": float(confidences[index]),
                "track_id": track_id,
            }
        )

    return candidates


def collect_dog_targets(
    video_path: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
    initial_reference_point: tuple[float, float],
) -> dict[int, dict]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"No se puede abrir el video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "FPS incorrectos durante el tracking del perro."
        )

    model = YOLO(
        str(model_path)
    )

    selector = DogTargetSelector()
    targets: dict[int, dict] = {}

    frame_index = 0
    frames_with_dog = 0
    max_dogs_in_frame = 0

    print("=" * 70)
    print("TRACKING DEL PERRO OBJETIVO")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}")
    print(f"Confianza minima: {confidence}")
    print()

    while True:
        success, frame = capture.read()

        if not success:
            break

        result = model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[1],
            conf=confidence,
            iou=0.5,
            imgsz=image_size,
            verbose=False,
        )[0]

        candidates = extract_dog_candidates(
            result
        )

        dogs_in_frame = len(candidates)

        if dogs_in_frame > 0:
            frames_with_dog += 1

        max_dogs_in_frame = max(
            max_dogs_in_frame,
            dogs_in_frame,
        )

        selected_dog = selector.select(
            candidates=candidates,
            fallback_point=initial_reference_point,
        )

        if selected_dog is not None:
            x1, y1, x2, y2 = selected_dog["box"]
            center_x, center_y = selected_dog["center"]

            targets[frame_index] = {
                "target_x": float(center_x),
                "target_y": float(center_y),
                "desired_x": float(center_x),
                "desired_y": float(center_y),
                "box_x1": float(x1),
                "box_y1": float(y1),
                "box_x2": float(x2),
                "box_y2": float(y2),
                "track_id": selected_dog["track_id"],
                "confidence": selected_dog["confidence"],
                "dogs_in_frame": dogs_in_frame,
                "status": (
                    "MULTI_DOG"
                    if dogs_in_frame > 1
                    else "DOG_DETECTED"
                ),
            }

        frame_index += 1

        if frame_index % 100 == 0:
            print(
                f"Procesados {frame_index}/{total_frames} frames"
            )

    capture.release()

    if not targets:
        raise RuntimeError(
            "No se ha detectado ningun perro valido en el video."
        )

    detection_percentage = (
        100.0 * frames_with_dog / frame_index
        if frame_index > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("TRACKING DEL PERRO FINALIZADO")
    print("=" * 70)
    print(f"Frames procesados: {frame_index}")
    print(f"Frames con perro: {frames_with_dog}")
    print(
        f"Porcentaje con deteccion: {detection_percentage:.2f}%"
    )
    print(
        f"Maximo de perros en un frame: {max_dogs_in_frame}"
    )
    print()

    return targets


def simulate_robot_mpc_dog(
    video_path: Path,
    model_path: Path,
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
    run_name: str,
) -> None:
    if horizon <= 0:
        raise ValueError(
            "El horizonte debe ser mayor que 0."
        )

    if control_period <= 0:
        raise ValueError(
            "control-period debe ser mayor que 0."
        )

    if position_scale <= 0:
        raise ValueError(
            "position-scale debe ser mayor que 0."
        )

    if stop_radius <= 0:
        raise ValueError(
            "stop-radius debe ser mayor que 0."
        )

    video_path = resolve_project_path(
        video_path
    )

    model_path = resolve_project_path(
        model_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"No existe el video: {video_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {model_path}"
        )

    guidance = collect_dog_targets(
        video_path=video_path,
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        initial_reference_point=(
            robot_start_x,
            robot_start_y,
        ),
    )

    guidance = smooth_guidance(
        guidance=guidance,
        alpha=smoothing_alpha,
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"No se puede abrir: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "FPS incorrectos."
        )

    dt_video = 1.0 / fps

    control_interval_frames = max(
        1,
        int(
            round(
                control_period * fps
            )
        ),
    )

    dt_control = (
        control_interval_frames / fps
    )

    horizon_seconds = (
        horizon * dt_control
    )

    max_omega = math.radians(
        max_omega_degrees
    )

    max_angular_acceleration = math.radians(
        max_angular_acceleration_degrees
    )

    output_directory = (
        OUTPUT_ROOT / run_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_directory
        / f"{video_path.stem}_dog_mpc.mp4"
    )

    output_csv = (
        output_directory
        / f"{video_path.stem}_dog_mpc.csv"
    )

    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(
            "No se ha podido crear el video."
        )

    state = np.array(
        [
            robot_start_x,
            robot_start_y,
            math.radians(
                initial_heading_degrees
            ),
            0.0,
            0.0,
        ],
        dtype=float,
    )

    current_control = np.array(
        [0.0, 0.0],
        dtype=float,
    )

    previous_solution = None

    robot_trajectory = deque(
        maxlen=500
    )

    last_target_data = None
    last_predicted_states = None
    last_reference_points = None
    last_mpc_cost = 0.0
    last_optimizer_success = True
    last_optimizer_iterations = 0

    next_control_frame = 0
    frame_index = 0
    optimizer_failures = 0
    optimizer_updates = 0

    print("=" * 70)
    print("SIMULACION MPC SEGUIMIENTO DE PERRO")
    print("=" * 70)
    print(f"FPS video: {fps:.3f}")
    print(f"Ts MPC real: {dt_control:.4f} s")
    print(f"Horizonte Hp: {horizon}")
    print(f"Horizonte temporal: {horizon_seconds:.3f} s")
    print(f"Escala posicion: {position_scale:.1f} px")
    print(f"Radio de frenado: {stop_radius:.1f} px")
    print(f"EMA alpha: {smoothing_alpha}")
    print()

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        csv_writer = csv.writer(
            csv_file
        )

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
                "reference_x",
                "reference_y",
                "distance_to_target",
                "target_track_id",
                "target_confidence",
                "dogs_in_frame",
                "target_status",
                "mpc_cost",
                "optimizer_success",
                "optimizer_iterations",
                "mpc_updated",
                "control_period_seconds",
                "horizon_seconds",
            ]
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            new_target_data = guidance.get(
                frame_index
            )

            if new_target_data is not None:
                last_target_data = new_target_data

            data = last_target_data
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

                    current_control = (
                        optimal_controls[0].copy()
                    )

                    previous_solution = (
                        optimal_controls.copy()
                    )

                    last_predicted_states = (
                        predicted_states
                    )

                    last_reference_points = (
                        reference_points
                    )

                    last_mpc_cost = mpc_cost
                    last_optimizer_success = optimizer_success
                    last_optimizer_iterations = optimizer_iterations

                    optimizer_updates += 1

                    if not optimizer_success:
                        optimizer_failures += 1

                    mpc_updated = True

                    next_control_frame = (
                        frame_index
                        + control_interval_frames
                    )

                state = update_state(
                    state=state,
                    control=current_control,
                    dt=dt_video,
                )

                state[0] = clamp(
                    state[0],
                    0.0,
                    width - 1.0,
                )

                state[1] = clamp(
                    state[1],
                    0.0,
                    height - 1.0,
                )

                robot_point = (
                    int(state[0]),
                    int(state[1]),
                )

                target_point = (
                    int(data["target_x"]),
                    int(data["target_y"]),
                )

                reference_point = (
                    int(data["desired_x"]),
                    int(data["desired_y"]),
                )

                distance_to_target = math.hypot(
                    data["target_x"] - state[0],
                    data["target_y"] - state[1],
                )

                robot_trajectory.append(
                    robot_point
                )

                x1 = int(round(data["box_x1"]))
                y1 = int(round(data["box_y1"]))
                x2 = int(round(data["box_x2"]))
                y2 = int(round(data["box_y2"]))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3,
                )

                cv2.circle(
                    frame,
                    target_point,
                    10,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    (
                        "PERRO OBJETIVO "
                        f"ID:{data['track_id']}"
                    ),
                    (
                        x1,
                        max(30, y1 - 15),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    reference_point,
                    14,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "REFERENCIA MPC",
                    (
                        reference_point[0] + 20,
                        reference_point[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                draw_reference_trajectory(
                    frame=frame,
                    reference_points=last_reference_points,
                )

                draw_predicted_trajectory(
                    frame=frame,
                    predicted_states=last_predicted_states,
                )

                if len(robot_trajectory) >= 2:
                    trajectory_points = np.array(
                        robot_trajectory,
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))

                    cv2.polylines(
                        frame,
                        [trajectory_points],
                        False,
                        (255, 0, 255),
                        3,
                    )

                draw_robot(
                    frame=frame,
                    robot_position=robot_point,
                    theta=state[2],
                )

                theta_degrees = math.degrees(
                    state[2]
                )

                omega_degrees = math.degrees(
                    current_control[1]
                )

                cv2.putText(
                    frame,
                    f"MPC Hp: {horizon}",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Ts MPC: {dt_control:.3f} s",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Horizonte: {horizon_seconds:.2f} s",
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"v: {current_control[0]:.1f} px/s",
                    (30, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"omega: {omega_degrees:.1f} deg/s",
                    (30, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Distancia perro: {distance_to_target:.1f} px",
                    (30, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Coste MPC: {last_mpc_cost:.5f}",
                    (30, 285),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                optimizer_text = (
                    "OK"
                    if last_optimizer_success
                    else "FALLO"
                )

                cv2.putText(
                    frame,
                    f"Optimizador: {optimizer_text}",
                    (30, 325),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Perros detectados: "
                        f"{int(data['dogs_in_frame'])}"
                    ),
                    (30, 365),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                if mpc_updated:
                    cv2.putText(
                        frame,
                        "MPC ACTUALIZADO",
                        (30, 405),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
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
                        f"{distance_to_target:.3f}",
                        data["track_id"],
                        f"{data['confidence']:.4f}",
                        int(data["dogs_in_frame"]),
                        data["status"],
                        f"{last_mpc_cost:.8f}",
                        last_optimizer_success,
                        last_optimizer_iterations,
                        mpc_updated,
                        f"{dt_control:.6f}",
                        f"{horizon_seconds:.6f}",
                    ]
                )

            video_writer.write(
                frame
            )

            frame_index += 1

            if frame_index % 250 == 0:
                print(
                    f"{frame_index}/{total_frames}"
                )

    capture.release()
    video_writer.release()

    print()
    print("=" * 70)
    print("SIMULACION MPC DE PERRO FINALIZADA")
    print("=" * 70)
    print(f"Frames procesados: {frame_index}")
    print(f"Actualizaciones MPC: {optimizer_updates}")
    print(f"Fallos optimizador: {optimizer_failures}")

    if optimizer_updates > 0:
        failure_percentage = (
            100.0
            * optimizer_failures
            / optimizer_updates
        )

        print(
            f"Porcentaje fallos: {failure_percentage:.2f}%"
        )

    print(f"Video: {output_video}")
    print(f"CSV: {output_csv}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MPC para seguir directamente al perro detectado."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--robot-start-x",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--robot-start-y",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--initial-heading",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-speed",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--max-omega",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=150.0,
    )

    parser.add_argument(
        "--max-angular-acceleration",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--control-period",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--position-scale",
        type=float,
        default=200.0,
    )

    parser.add_argument(
        "--stop-radius",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--weight-tracking",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--weight-energy",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--weight-terminal-position",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--weight-terminal-velocity",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--weight-smoothness",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help=(
            "1.0 sigue el centro del perro sin suavizado adicional."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="mpc_dog_target",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    simulate_robot_mpc_dog(
        video_path=args.video,
        model_path=args.model,
        robot_start_x=args.robot_start_x,
        robot_start_y=args.robot_start_y,
        initial_heading_degrees=args.initial_heading,
        max_speed=args.max_speed,
        max_omega_degrees=args.max_omega,
        max_acceleration=args.max_acceleration,
        max_angular_acceleration_degrees=(
            args.max_angular_acceleration
        ),
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
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
