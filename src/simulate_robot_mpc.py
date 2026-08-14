import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "robot_mpc"


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

    Para cada frame tenemos:
      - centro del rebaño
      - destino global
      - punto deseado del robot
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


def update_state(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Modelo cinemático discreto.

    Estado:
        [x, y, theta, v, omega]

    Control:
        [v, omega]

    Basado en:

        x(k+1) = x(k) + v cos(theta) Ts
        y(k+1) = y(k) + v sin(theta) Ts
        theta(k+1) = theta(k) + omega Ts
    """

    x, y, theta, _, _ = state

    v = control[0]
    omega = control[1]

    new_x = (
        x
        + v * math.cos(theta) * dt
    )

    new_y = (
        y
        + v * math.sin(theta) * dt
    )

    new_theta = normalize_angle(
        theta + omega * dt
    )

    return np.array(
        [
            new_x,
            new_y,
            new_theta,
            v,
            omega,
        ],
        dtype=float,
    )


def predict_states(
    initial_state: np.ndarray,
    controls: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Predice la trayectoria futura del robot para una
    secuencia de controles.

    controls tiene forma:

        [
            [v0, omega0],
            [v1, omega1],
            ...
        ]
    """

    state = initial_state.copy()

    predicted_states = []

    for control in controls:
        state = update_state(
            state=state,
            control=control,
            dt=dt,
        )

        predicted_states.append(
            state.copy()
        )

    return np.array(
        predicted_states
    )


def calculate_mpc_cost(
    flat_controls: np.ndarray,
    initial_state: np.ndarray,
    desired_point: tuple[float, float],
    horizon: int,
    dt: float,
    image_diagonal: float,
    max_speed: float,
    max_omega: float,
    weight_tracking: float,
    weight_energy: float,
) -> float:
    """
    Función objetivo del MPC.

    Tiene tres componentes:

    1. Error de seguimiento durante todo el horizonte.
    2. Error terminal: damos mucha importancia a dónde
       termina el robot al final del horizonte.
    3. Coste de control: penaliza velocidades y giros
       excesivos.
    """

    controls = flat_controls.reshape(
        horizon,
        2,
    )

    predicted_states = predict_states(
        initial_state=initial_state,
        controls=controls,
        dt=dt,
    )

    desired_x, desired_y = desired_point

    tracking_cost = 0.0
    energy_cost = 0.0

    epsilon = 1e-9

    normalization = (
        image_diagonal * image_diagonal
        + epsilon
    )

    # ==========================================
    # COSTE DE TODA LA TRAYECTORIA
    # ==========================================

    for index in range(horizon):
        predicted_x = predicted_states[index, 0]
        predicted_y = predicted_states[index, 1]

        dx = predicted_x - desired_x
        dy = predicted_y - desired_y

        squared_distance = (
            dx * dx
            + dy * dy
        )

        # Los estados más lejanos del horizonte
        # reciben mayor importancia.
        progress_weight = (
            index + 1
        ) / horizon

        tracking_cost += (
            progress_weight
            * squared_distance
            / normalization
        )

        v = controls[index, 0]
        omega = controls[index, 1]

        normalized_v = (
            v
            / (max_speed + epsilon)
        )

        normalized_omega = (
            omega
            / (max_omega + epsilon)
        )

        energy_cost += (
            normalized_v ** 2
            + normalized_omega ** 2
        )

    # ==========================================
    # COSTE TERMINAL
    # ==========================================

    final_x = predicted_states[-1, 0]
    final_y = predicted_states[-1, 1]

    final_dx = (
        final_x - desired_x
    )

    final_dy = (
        final_y - desired_y
    )

    terminal_cost = (
        final_dx * final_dx
        + final_dy * final_dy
    ) / normalization

    # El punto final debe importar mucho.
    terminal_weight = 10.0

    total_cost = (
        weight_tracking * tracking_cost
        + terminal_weight * terminal_cost
        + weight_energy * energy_cost
    )

    return float(total_cost)


def calculate_rate_constraints(
    flat_controls: np.ndarray,
    previous_control: np.ndarray,
    horizon: int,
    dt: float,
    max_acceleration: float,
    max_angular_acceleration: float,
) -> np.ndarray:
    """
    Implementa las restricciones:

        |v(k) - v(k-1)|
            <= max_acceleration * Ts

        |omega(k) - omega(k-1)|
            <= max_angular_acceleration * Ts

    scipy considera válida una restricción cuando:

        constraint >= 0
    """

    controls = flat_controls.reshape(
        horizon,
        2,
    )

    maximum_delta_v = (
        max_acceleration
        * dt
    )

    maximum_delta_omega = (
        max_angular_acceleration
        * dt
    )

    constraints = []

    previous_v = previous_control[0]
    previous_omega = previous_control[1]

    for index in range(horizon):
        current_v = controls[index, 0]
        current_omega = controls[index, 1]

        delta_v = (
            current_v
            - previous_v
        )

        delta_omega = (
            current_omega
            - previous_omega
        )

        # +dv <= límite
        constraints.append(
            maximum_delta_v
            - delta_v
        )

        # -dv <= límite
        constraints.append(
            maximum_delta_v
            + delta_v
        )

        # +domega <= límite
        constraints.append(
            maximum_delta_omega
            - delta_omega
        )

        # -domega <= límite
        constraints.append(
            maximum_delta_omega
            + delta_omega
        )

        previous_v = current_v
        previous_omega = current_omega

    return np.array(
        constraints,
        dtype=float,
    )


def build_initial_guess(
    previous_solution: np.ndarray | None,
    previous_control: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Genera la solución inicial del optimizador.

    Si ya tenemos una solución del frame anterior,
    hacemos warm start desplazándola un paso.
    """

    if previous_solution is None:
        controls = np.zeros(
            (horizon, 2),
            dtype=float,
        )

        # Generamos una aceleración inicial progresiva.
        for index in range(horizon):
            controls[index, 0] = (
                previous_control[0]
                + (index + 1) * 2.0
            )

            controls[index, 1] = (
                previous_control[1]
            )

        return controls.flatten()

    shifted = np.vstack(
        [
            previous_solution[1:],
            previous_solution[-1],
        ]
    )

    return shifted.flatten()


def solve_mpc(
    state: np.ndarray,
    desired_point: tuple[float, float],
    previous_control: np.ndarray,
    previous_solution: np.ndarray | None,
    horizon: int,
    dt: float,
    image_diagonal: float,
    max_speed: float,
    max_omega: float,
    max_acceleration: float,
    max_angular_acceleration: float,
    weight_tracking: float,
    weight_energy: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    bool,
    int,
]:
    """
    Resuelve el problema MPC.

    Devuelve:
      - secuencia óptima de controles
      - trayectoria predicha
      - coste
      - éxito del optimizador
      - número de iteraciones
    """

    initial_guess = build_initial_guess(
        previous_solution=previous_solution,
        previous_control=previous_control,
        horizon=horizon,
    )

    bounds = []

    for _ in range(horizon):
        bounds.append(
            (
                0.0,
                max_speed,
            )
        )

        bounds.append(
            (
                -max_omega,
                max_omega,
            )
        )

    constraints = {
        "type": "ineq",
        "fun": calculate_rate_constraints,
        "args": (
            previous_control,
            horizon,
            dt,
            max_acceleration,
            max_angular_acceleration,
        ),
    }

    result = minimize(
        fun=calculate_mpc_cost,
        x0=initial_guess,
        args=(
            state,
            desired_point,
            horizon,
            dt,
            image_diagonal,
            max_speed,
            max_omega,
            weight_tracking,
            weight_energy,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 60,
            "ftol": 1e-6,
            "disp": False,
        },
    )

    if result.success:
        optimal_controls = (
            result.x.reshape(
                horizon,
                2,
            )
        )
    else:
        # Si el optimizador falla usamos
        # la mejor solución disponible.
        optimal_controls = (
            result.x.reshape(
                horizon,
                2,
            )
        )

    predicted_states = predict_states(
        initial_state=state,
        controls=optimal_controls,
        dt=dt,
    )

    return (
        optimal_controls,
        predicted_states,
        float(result.fun),
        bool(result.success),
        int(result.nit),
    )


def draw_robot(
    frame,
    robot_position: tuple[int, int],
    theta: float,
) -> None:

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
        "ROBOT MPC",
        (
            robot_x + 20,
            robot_y - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_predicted_trajectory(
    frame,
    predicted_states: np.ndarray,
) -> None:
    """
    Dibuja dónde predice el MPC que estará el robot
    durante los siguientes Hp pasos.
    """

    if len(predicted_states) < 2:
        return

    points = []

    for state in predicted_states:
        points.append(
            [
                int(state[0]),
                int(state[1]),
            ]
        )

    points = np.array(
        points,
        dtype=np.int32,
    ).reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        frame,
        [points],
        False,
        (0, 255, 255),
        3,
    )


def simulate_robot_mpc(
    video_path: Path,
    guidance_path: Path,
    robot_start_x: float,
    robot_start_y: float,
    initial_heading_degrees: float,
    max_speed: float,
    max_omega_degrees: float,
    max_acceleration: float,
    max_angular_acceleration_degrees: float,
    horizon: int,
    weight_tracking: float,
    weight_energy: float,
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

    dt = 1.0 / fps

    image_diagonal = math.hypot(
        width,
        height,
    )

    max_omega = math.radians(
        max_omega_degrees
    )

    max_angular_acceleration = math.radians(
        max_angular_acceleration_degrees
    )

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
        / f"{video_path.stem}_mpc.mp4"
    )

    output_csv = (
        output_directory
        / f"{video_path.stem}_mpc.csv"
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

    previous_control = np.array(
        [
            0.0,
            0.0,
        ],
        dtype=float,
    )

    previous_solution = None

    robot_trajectory = deque(
        maxlen=500
    )

    frame_index = 0

    optimizer_failures = 0

    print("=" * 70)
    print("SIMULACIÓN MPC")
    print("=" * 70)

    print(
        f"Horizonte Hp: {horizon}"
    )

    print(
        f"Peso seguimiento ws: "
        f"{weight_tracking}"
    )

    print(
        f"Peso energía wu: "
        f"{weight_energy}"
    )

    print(
        f"Ts: {dt:.4f} s"
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
                "desired_x",
                "desired_y",
                "mpc_cost",
                "optimizer_success",
                "optimizer_iterations",
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

                desired_point = (
                    data["desired_x"],
                    data["desired_y"],
                )

                (
                    optimal_controls,
                    predicted_states,
                    mpc_cost,
                    optimizer_success,
                    optimizer_iterations,
                ) = solve_mpc(
                    state=state,
                    desired_point=desired_point,
                    previous_control=previous_control,
                    previous_solution=previous_solution,
                    horizon=horizon,
                    dt=dt,
                    image_diagonal=image_diagonal,
                    max_speed=max_speed,
                    max_omega=max_omega,
                    max_acceleration=max_acceleration,
                    max_angular_acceleration=max_angular_acceleration,
                    weight_tracking=weight_tracking,
                    weight_energy=weight_energy,
                )

                if not optimizer_success:
                    optimizer_failures += 1

                # ========================================
                # PRINCIPIO FUNDAMENTAL DEL MPC:
                #
                # calculamos Hp controles,
                # pero aplicamos SOLO EL PRIMERO.
                # ========================================

                control = (
                    optimal_controls[0]
                )

                state = update_state(
                    state=state,
                    control=control,
                    dt=dt,
                )

                state[0] = max(
                    0,
                    min(
                        width - 1,
                        state[0],
                    ),
                )

                state[1] = max(
                    0,
                    min(
                        height - 1,
                        state[1],
                    ),
                )

                previous_control = (
                    control.copy()
                )

                previous_solution = (
                    optimal_controls.copy()
                )

                robot_point = (
                    int(state[0]),
                    int(state[1]),
                )

                desired_point_int = (
                    int(desired_point[0]),
                    int(desired_point[1]),
                )

                flock_point = (
                    int(data["flock_x"]),
                    int(data["flock_y"]),
                )

                robot_trajectory.append(
                    robot_point
                )

                # Centro rebaño
                cv2.circle(
                    frame,
                    flock_point,
                    10,
                    (255, 255, 0),
                    -1,
                )

                # Punto de conducción
                cv2.circle(
                    frame,
                    desired_point_int,
                    14,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "OBJETIVO MPC",
                    (
                        desired_point_int[0] + 20,
                        desired_point_int[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Robot
                draw_robot(
                    frame=frame,
                    robot_position=robot_point,
                    theta=state[2],
                )

                # Predicción futura MPC
                draw_predicted_trajectory(
                    frame=frame,
                    predicted_states=predicted_states,
                )

                # Trayectoria real
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

                theta_degrees = (
                    math.degrees(
                        state[2]
                    )
                )

                omega_degrees = (
                    math.degrees(
                        control[1]
                    )
                )

                cv2.putText(
                    frame,
                    f"MPC Hp: {horizon}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"v: "
                        f"{control[0]:.1f} px/s"
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
                        f"omega: "
                        f"{omega_degrees:.1f} deg/s"
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
                        f"Coste MPC: "
                        f"{mpc_cost:.5f}"
                    ),
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                status_text = (
                    "OK"
                    if optimizer_success
                    else "FALLO"
                )

                cv2.putText(
                    frame,
                    (
                        f"Optimizador: "
                        f"{status_text}"
                    ),
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{frame_index / fps:.3f}",
                        f"{state[0]:.3f}",
                        f"{state[1]:.3f}",
                        f"{theta_degrees:.3f}",
                        f"{control[0]:.3f}",
                        f"{omega_degrees:.3f}",
                        f"{desired_point[0]:.3f}",
                        f"{desired_point[1]:.3f}",
                        f"{mpc_cost:.8f}",
                        optimizer_success,
                        optimizer_iterations,
                    ]
                )

            video_writer.write(
                frame
            )

            frame_index += 1

            if frame_index % 100 == 0:
                print(
                    f"{frame_index}/{total_frames}"
                )

    capture.release()
    video_writer.release()

    print("\n" + "=" * 70)
    print("SIMULACIÓN MPC FINALIZADA")
    print("=" * 70)

    print(
        f"Frames: {frame_index}"
    )

    print(
        f"Fallos optimizador: "
        f"{optimizer_failures}"
    )

    print(
        f"Vídeo: {output_video}"
    )

    print(
        f"CSV: {output_csv}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "MPC para el seguimiento del "
            "punto de conducción."
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
        "--weight-tracking",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--weight-energy",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="mpc_v1",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    simulate_robot_mpc(
        video_path=args.video,
        guidance_path=args.guidance,
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
        weight_tracking=args.weight_tracking,
        weight_energy=args.weight_energy,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()