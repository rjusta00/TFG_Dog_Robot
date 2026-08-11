import argparse
import csv
import math
from pathlib import Path
from collections import deque

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "robot_kinematics"


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def normalize_angle(angle: float) -> float:
    """
    Normaliza un ángulo al intervalo [-pi, pi].
    """

    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def load_guidance(
    guidance_path: Path,
) -> dict[int, dict]:
    """
    Carga el CSV generado por calculate_robot_guidance.py.

    Para cada frame obtenemos:
      - centro del rebaño
      - posición deseada del robot
      - destino
    """

    if not guidance_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV: {guidance_path}"
        )

    guidance = {}

    with guidance_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            frame_index = int(row["frame"])

            if (
                not row["desired_robot_x"]
                or not row["desired_robot_y"]
            ):
                continue

            guidance[frame_index] = {
                "flock_x": float(row["flock_center_x"]),
                "flock_y": float(row["flock_center_y"]),
                "target_x": float(row["target_x"]),
                "target_y": float(row["target_y"]),
                "desired_x": float(row["desired_robot_x"]),
                "desired_y": float(row["desired_robot_y"]),
                "status": row["status"],
            }

    if not guidance:
        raise RuntimeError(
            "El CSV no contiene posiciones de guiado válidas."
        )

    return guidance


def calculate_desired_heading(
    robot_x: float,
    robot_y: float,
    target_x: float,
    target_y: float,
) -> float:
    """
    Calcula el ángulo que debería tener el robot para mirar
    hacia el punto deseado.

    OpenCV utiliza:
       X positivo -> derecha
       Y positivo -> abajo

    Por eso usamos atan2(dy, dx) directamente para nuestra
    simulación en coordenadas de imagen.
    """

    dx = target_x - robot_x
    dy = target_y - robot_y

    return math.atan2(
        dy,
        dx,
    )


def calculate_desired_control(
    robot_x: float,
    robot_y: float,
    theta: float,
    desired_x: float,
    desired_y: float,
    max_speed: float,
    max_omega: float,
    heading_gain: float,
    arrival_radius: float,
) -> tuple[float, float, float, float]:
    """
    Calcula el control que querríamos aplicar al robot.

    Todavía NO tiene en cuenta aceleraciones.
    Devuelve:

        desired_v
        desired_omega
        distance
        heading_error
    """

    dx = desired_x - robot_x
    dy = desired_y - robot_y

    distance = math.hypot(
        dx,
        dy,
    )

    # Si estamos suficientemente cerca,
    # queremos detenernos.
    if distance <= arrival_radius:
        return (
            0.0,
            0.0,
            distance,
            0.0,
        )

    desired_heading = calculate_desired_heading(
        robot_x=robot_x,
        robot_y=robot_y,
        target_x=desired_x,
        target_y=desired_y,
    )

    heading_error = normalize_angle(
        desired_heading - theta
    )

    # ===============================
    # VELOCIDAD ANGULAR DESEADA
    # ===============================

    desired_omega = (
        heading_gain
        * heading_error
    )

    desired_omega = max(
        -max_omega,
        min(
            max_omega,
            desired_omega,
        ),
    )

    # ===============================
    # VELOCIDAD LINEAL DESEADA
    # ===============================

    # Si el robot está mal orientado,
    # reducimos la velocidad.
    alignment = max(
        0.0,
        math.cos(heading_error),
    )

    desired_v = (
        max_speed
        * alignment
    )

    # Reducimos velocidad al acercarnos
    # al punto deseado.
    slowdown_distance = 150.0

    if distance < slowdown_distance:
        desired_v *= (
            distance
            / slowdown_distance
        )

    return (
        desired_v,
        desired_omega,
        distance,
        heading_error,
    )

def limit_rate(
    desired_value: float,
    previous_value: float,
    maximum_rate: float,
    dt: float,
) -> float:
    """
    Limita cuánto puede cambiar una variable entre dos instantes.

    Implementa:

        |u(k) - u(k-1)| <= u_dot_max * Ts
    """

    maximum_change = (
        maximum_rate
        * dt
    )

    difference = (
        desired_value
        - previous_value
    )

    difference = max(
        -maximum_change,
        min(
            maximum_change,
            difference,
        ),
    )

    return (
        previous_value
        + difference
    )

def apply_control_constraints(
    desired_v: float,
    desired_omega: float,
    previous_v: float,
    previous_omega: float,
    dt: float,
    min_speed: float,
    max_speed: float,
    max_omega: float,
    max_acceleration: float,
    max_angular_acceleration: float,
) -> tuple[float, float]:
    """
    Aplica las restricciones físicas del robot.

    1. Límite de velocidad.
    2. Límite de velocidad angular.
    3. Límite de aceleración lineal.
    4. Límite de aceleración angular.
    """

    # ====================================
    # 1. Límites absolutos
    # ====================================

    desired_v = max(
        min_speed,
        min(
            max_speed,
            desired_v,
        ),
    )

    desired_omega = max(
        -max_omega,
        min(
            max_omega,
            desired_omega,
        ),
    )

    # ====================================
    # 2. Restricción de aceleración
    # ====================================

    constrained_v = limit_rate(
        desired_value=desired_v,
        previous_value=previous_v,
        maximum_rate=max_acceleration,
        dt=dt,
    )

    # ====================================
    # 3. Restricción de aceleración angular
    # ====================================

    constrained_omega = limit_rate(
        desired_value=desired_omega,
        previous_value=previous_omega,
        maximum_rate=max_angular_acceleration,
        dt=dt,
    )

    return (
        constrained_v,
        constrained_omega,
    )


def update_robot_state(
    x: float,
    y: float,
    theta: float,
    v: float,
    omega: float,
    dt: float,
) -> tuple[float, float, float]:
    """
    Modelo cinemático discreto.

    x(k+1) = x(k) + v cos(theta) Ts
    y(k+1) = y(k) + v sin(theta) Ts
    theta(k+1) = theta(k) + omega Ts
    """

    new_x = (
        x
        + v
        * math.cos(theta)
        * dt
    )

    new_y = (
        y
        + v
        * math.sin(theta)
        * dt
    )

    new_theta = (
        theta
        + omega
        * dt
    )

    new_theta = normalize_angle(
        new_theta
    )

    return (
        new_x,
        new_y,
        new_theta,
    )


def draw_robot(
    frame,
    robot_position: tuple[int, int],
    theta: float,
) -> None:
    """
    Dibuja el robot y su orientación.
    """

    robot_x, robot_y = robot_position

    cv2.circle(
        frame,
        robot_position,
        16,
        (255, 0, 255),
        -1,
    )

    heading_length = 55

    heading_x = int(
        robot_x
        + heading_length
        * math.cos(theta)
    )

    heading_y = int(
        robot_y
        + heading_length
        * math.sin(theta)
    )

    cv2.arrowedLine(
        frame,
        robot_position,
        (
            heading_x,
            heading_y,
        ),
        (255, 0, 255),
        5,
        tipLength=0.25,
    )

    cv2.putText(
        frame,
        "ROBOT",
        (
            robot_x + 20,
            robot_y - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def simulate_robot(
    video_path: Path,
    guidance_path: Path,
    robot_start_x: float,
    robot_start_y: float,
    initial_heading_degrees: float,
    max_speed: float,
    max_omega_degrees: float,
    max_acceleration: float,
    max_angular_acceleration_degrees: float,
    heading_gain: float,
    arrival_radius: float,
    run_name: str,
) -> None:

    video_path = resolve_project_path(
        video_path
    )

    guidance_path = resolve_project_path(
        guidance_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"No existe el vídeo: {video_path}"
        )

    guidance = load_guidance(
        guidance_path
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

    # Periodo de muestreo Ts.
    dt = 1.0 / fps

    output_directory = (
        OUTPUT_ROOT
        / run_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_directory
        / f"{video_path.stem}_kinematics.mp4"
    )

    output_csv = (
        output_directory
        / f"{video_path.stem}_kinematics.csv"
    )

    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            width,
            height,
        ),
    )

    if not video_writer.isOpened():
        capture.release()

        raise RuntimeError(
            "No se ha podido crear el vídeo."
        )

    # ==========================
    # ESTADO INICIAL DEL ROBOT
    # ==========================

    robot_x = robot_start_x
    robot_y = robot_start_y

    theta = math.radians(
        initial_heading_degrees
    )

    v = 0.0
    omega = 0.0

    max_omega = math.radians(
        max_omega_degrees
    )

    max_angular_acceleration = math.radians(
        max_angular_acceleration_degrees
    )

    robot_trajectory = deque(maxlen=500)

    frame_index = 0

    print("=" * 70)
    print("SIMULACIÓN CINEMÁTICA DEL ROBOT")
    print("=" * 70)

    print(
        f"Posición inicial: "
        f"({robot_x}, {robot_y})"
    )

    print(
        f"Orientación inicial: "
        f"{initial_heading_degrees} grados"
    )

    print(
        f"Velocidad máxima: "
        f"{max_speed:.2f} px/s"
    )

    print(
        f"Velocidad angular máxima: "
        f"{max_omega_degrees:.2f} grados/s"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Ts: {dt:.4f} segundos"
    )

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
                "linear_acceleration",
                "angular_acceleration_degrees",
                "desired_x",
                "desired_y",
                "distance",
                "heading_error_degrees",
            ]
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            data = guidance.get(
                frame_index
            )

            if data is not None:

                desired_x = (
                    data["desired_x"]
                )

                desired_y = (
                    data["desired_y"]
                )

                (
                    desired_v,
                    desired_omega,
                    distance,
                    heading_error,
                ) = calculate_desired_control(
                    robot_x=robot_x,
                    robot_y=robot_y,
                    theta=theta,
                    desired_x=desired_x,
                    desired_y=desired_y,
                    max_speed=max_speed,
                    max_omega=max_omega,
                    heading_gain=heading_gain,
                    arrival_radius=arrival_radius,
                )

                previous_v = v
                previous_omega = omega

                (
                    v,
                    omega,
                ) = apply_control_constraints(
                    desired_v=desired_v,
                    desired_omega=desired_omega,
                    previous_v=previous_v,
                    previous_omega=previous_omega,
                    dt=dt,
                    min_speed=0.0,
                    max_speed=max_speed,
                    max_omega=max_omega,
                    max_acceleration=max_acceleration,
                    max_angular_acceleration=max_angular_acceleration,
                )

                # Aplicamos la ecuación cinemática.
                (
                    robot_x,
                    robot_y,
                    theta,
                ) = update_robot_state(
                    x=robot_x,
                    y=robot_y,
                    theta=theta,
                    v=v,
                    omega=omega,
                    dt=dt,
                )

                robot_x = max(
                    0,
                    min(width - 1, robot_x),
                )

                robot_y = max(
                    0,
                    min(height - 1, robot_y),
                )

                robot_point = (
                    int(robot_x),
                    int(robot_y),
                )

                desired_point = (
                    int(desired_x),
                    int(desired_y),
                )

                flock_point = (
                    int(data["flock_x"]),
                    int(data["flock_y"]),
                )

                robot_trajectory.append(
                    robot_point
                )

                # Centro del rebaño.
                cv2.circle(
                    frame,
                    flock_point,
                    10,
                    (255, 255, 0),
                    -1,
                )

                # Punto deseado del robot.
                cv2.circle(
                    frame,
                    desired_point,
                    14,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "OBJETIVO ROBOT",
                    (
                        desired_point[0] + 20,
                        desired_point[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Robot y orientación.
                draw_robot(
                    frame=frame,
                    robot_position=robot_point,
                    theta=theta,
                )

                # Línea al objetivo.
                cv2.line(
                    frame,
                    robot_point,
                    desired_point,
                    (0, 0, 255),
                    2,
                )

                # Trayectoria.
                if len(robot_trajectory) >= 2:

                    points = np.array(
                        robot_trajectory,
                        dtype=np.int32,
                    ).reshape(
                        (-1, 1, 2)
                    )

                    cv2.polylines(
                        frame,
                        [points],
                        False,
                        (255, 0, 255),
                        3,
                    )

                theta_degrees = math.degrees(
                    theta
                )

                omega_degrees = math.degrees(
                    omega
                )

                heading_error_degrees = (
                    math.degrees(
                        heading_error
                    )
                )

                linear_acceleration = (
                    v - previous_v
                ) / dt

                angular_acceleration = (
                    omega - previous_omega
                ) / dt

                cv2.putText(
                    frame,
                    (
                        f"Distancia: "
                        f"{distance:.1f} px"
                    ),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Velocidad: "
                        f"{v:.1f} px/s"
                    ),
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Orientacion: "
                        f"{theta_degrees:.1f} deg"
                    ),
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Omega: "
                        f"{omega_degrees:.1f} deg/s"
                    ),
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Aceleracion: "
                        f"{linear_acceleration:.1f} px/s2"
                    ),
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"Acel. angular: "
                        f"{math.degrees(angular_acceleration):.1f} deg/s2"
                    ),
                    (30, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{frame_index / fps:.3f}",
                        f"{robot_x:.3f}",
                        f"{robot_y:.3f}",
                        f"{theta_degrees:.3f}",
                        f"{v:.3f}",
                        f"{omega_degrees:.3f}",
                        f"{linear_acceleration:.3f}",
                        f"{math.degrees(angular_acceleration):.3f}",
                        f"{desired_x:.3f}",
                        f"{desired_y:.3f}",
                        f"{distance:.3f}",
                        f"{heading_error_degrees:.3f}",
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

    print("\n" + "=" * 70)
    print("SIMULACIÓN FINALIZADA")
    print("=" * 70)

    print(
        f"Vídeo: {output_video}"
    )

    print(
        f"CSV: {output_csv}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Simulación cinemática del robot "
            "siguiendo el punto de conducción."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--guidance",
        type=Path,
        required=True,
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
        help="Orientación inicial en grados.",
    )

    parser.add_argument(
        "--max-speed",
        type=float,
        default=250.0,
        help="Velocidad máxima en píxeles/segundo.",
    )

    parser.add_argument(
        "--max-omega",
        type=float,
        default=120.0,
        help=(
            "Velocidad angular máxima "
            "en grados/segundo."
        ),
    )

    parser.add_argument(
        "--heading-gain",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--arrival-radius",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="kinematics_v1",
    )

    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=150.0,
        help=(
            "Aceleración lineal máxima "
            "en píxeles/segundo²."
        ),
    )

    parser.add_argument(
        "--max-angular-acceleration",
        type=float,
        default=180.0,
        help=(
            "Aceleración angular máxima "
            "en grados/segundo²."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    simulate_robot(
        video_path=args.video,
        guidance_path=args.guidance,
        robot_start_x=args.robot_start_x,
        robot_start_y=args.robot_start_y,
        initial_heading_degrees=args.initial_heading,
        max_speed=args.max_speed,
        max_omega_degrees=args.max_omega,
        heading_gain=args.heading_gain,
        arrival_radius=args.arrival_radius,
        run_name=args.name,
        max_acceleration=args.max_acceleration,
        max_angular_acceleration_degrees=args.max_angular_acceleration,
    )


if __name__ == "__main__":
    main()