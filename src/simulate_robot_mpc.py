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


# ============================================================
# UTILIDADES
# ============================================================


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def normalize_angle(angle: float) -> float:
    """
    Normaliza un ángulo al intervalo [-pi, pi].
    """

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def move_towards(
    current: float,
    target: float,
    maximum_change: float,
) -> float:
    """
    Mueve current hacia target sin superar
    maximum_change en una única actualización.
    """

    difference = (
        target
        - current
    )

    difference = clamp(
        difference,
        -maximum_change,
        maximum_change,
    )

    return (
        current
        + difference
    )


# ============================================================
# CARGA DEL GUIADO
# ============================================================


def load_guidance(
    guidance_path: Path,
) -> dict[int, dict]:
    """
    Carga el CSV generado por calculate_robot_guidance.py.
    """

    if not guidance_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV: {guidance_path}"
        )

    guidance: dict[int, dict] = {}

    with guidance_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            if not row.get("frame"):
                continue

            if (
                not row.get("desired_robot_x")
                or not row.get("desired_robot_y")
            ):
                continue

            frame_index = int(
                row["frame"]
            )

            guidance[frame_index] = {
                "flock_x": float(
                    row["flock_center_x"]
                ),
                "flock_y": float(
                    row["flock_center_y"]
                ),
                "target_x": float(
                    row["target_x"]
                ),
                "target_y": float(
                    row["target_y"]
                ),
                "desired_x": float(
                    row["desired_robot_x"]
                ),
                "desired_y": float(
                    row["desired_robot_y"]
                ),
                "status": row.get(
                    "status",
                    "",
                ),
            }

    if not guidance:
        raise RuntimeError(
            "El CSV no contiene posiciones de guiado válidas."
        )

    return guidance


# ============================================================
# SUAVIZADO EMA
# ============================================================


def smooth_guidance(
    guidance: dict[int, dict],
    alpha: float,
) -> dict[int, dict]:
    """
    Suaviza el punto de conducción usando una
    media móvil exponencial.

    alpha pequeño:
        más suavizado.

    alpha cercano a 1:
        menos suavizado.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            "smoothing-alpha debe estar en (0, 1]."
        )

    smoothed: dict[int, dict] = {}

    previous_x = None
    previous_y = None

    for frame_index in sorted(
        guidance.keys()
    ):

        data = (
            guidance[
                frame_index
            ].copy()
        )

        current_x = (
            data["desired_x"]
        )

        current_y = (
            data["desired_y"]
        )

        if previous_x is None:

            filtered_x = (
                current_x
            )

            filtered_y = (
                current_y
            )

        else:

            filtered_x = (
                alpha * current_x
                + (1.0 - alpha)
                * previous_x
            )

            filtered_y = (
                alpha * current_y
                + (1.0 - alpha)
                * previous_y
            )

        data["desired_x"] = (
            filtered_x
        )

        data["desired_y"] = (
            filtered_y
        )

        smoothed[
            frame_index
        ] = data

        previous_x = (
            filtered_x
        )

        previous_y = (
            filtered_y
        )

    return smoothed


# ============================================================
# HORIZONTE FUTURO DE REFERENCIA
# ============================================================


def build_reference_horizon(
    guidance: dict[int, dict],
    frame_index: int,
    current_data: dict,
    horizon: int,
    control_interval_frames: int,
) -> np.ndarray:
    """
    Construye:

        s(k+1)
        s(k+2)
        ...
        s(k+Hp)

    teniendo en cuenta el periodo real del MPC.

    Ejemplo:

        vídeo = 30 FPS
        Ts MPC = 0.1 s

    aproximadamente:

        frame + 3
        frame + 6
        frame + 9
        ...
    """

    references = []

    last_x = (
        current_data[
            "desired_x"
        ]
    )

    last_y = (
        current_data[
            "desired_y"
        ]
    )

    for step in range(
        1,
        horizon + 1,
    ):

        future_frame = (
            frame_index
            + step
            * control_interval_frames
        )

        future_data = (
            guidance.get(
                future_frame
            )
        )

        if future_data is not None:

            last_x = (
                future_data[
                    "desired_x"
                ]
            )

            last_y = (
                future_data[
                    "desired_y"
                ]
            )

        references.append(
            [
                last_x,
                last_y,
            ]
        )

    return np.array(
        references,
        dtype=float,
    )


# ============================================================
# MODELO CINEMÁTICO
# ============================================================


def update_state(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Estado:

        [x, y, theta, v, omega]

    Control:

        [v, omega]

    Modelo:

        x(k+1) =
            x(k) + v cos(theta) Ts

        y(k+1) =
            y(k) + v sin(theta) Ts

        theta(k+1) =
            theta(k) + omega Ts
    """

    x, y, theta, _, _ = (
        state
    )

    v = float(
        control[0]
    )

    omega = float(
        control[1]
    )

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
        normalize_angle(
            theta
            + omega * dt
        )
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
    Predice los siguientes Hp estados.
    """

    state = (
        initial_state.copy()
    )

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
        predicted_states,
        dtype=float,
    )


# ============================================================
# FUNCIÓN DE COSTE MPC
# ============================================================


def calculate_mpc_cost(
    flat_controls: np.ndarray,
    initial_state: np.ndarray,
    reference_points: np.ndarray,
    horizon: int,
    dt: float,
    position_scale: float,
    stop_radius: float,
    max_speed: float,
    max_omega: float,
    weight_tracking: float,
    weight_energy: float,
    weight_terminal_position: float,
    weight_terminal_velocity: float,
    weight_smoothness: float,
) -> float:
    """
    Función objetivo MPC.

    J =
        seguimiento
        + energía
        + posición terminal
        + velocidad terminal
        + suavidad

    Cambio importante:

    La velocidad terminal solo se penaliza
    cuando el robot está cerca del objetivo.
    """

    controls = (
        flat_controls.reshape(
            horizon,
            2,
        )
    )

    predicted_states = (
        predict_states(
            initial_state=initial_state,
            controls=controls,
            dt=dt,
        )
    )

    epsilon = 1e-9

    # ========================================================
    # NORMALIZACIÓN DE POSICIÓN
    #
    # Ya NO usamos la diagonal completa de la imagen.
    # ========================================================

    position_normalization = (
        position_scale ** 2
        + epsilon
    )

    tracking_cost = 0.0
    energy_cost = 0.0
    smoothness_cost = 0.0

    previous_v = (
        initial_state[3]
    )

    previous_omega = (
        initial_state[4]
    )

    # ========================================================
    # COSTE DURANTE TODO EL HORIZONTE
    # ========================================================

    for index in range(
        horizon
    ):

        predicted_x = (
            predicted_states[
                index,
                0,
            ]
        )

        predicted_y = (
            predicted_states[
                index,
                1,
            ]
        )

        reference_x = (
            reference_points[
                index,
                0,
            ]
        )

        reference_y = (
            reference_points[
                index,
                1,
            ]
        )

        dx = (
            predicted_x
            - reference_x
        )

        dy = (
            predicted_y
            - reference_y
        )

        squared_distance = (
            dx * dx
            + dy * dy
        )

        progress_weight = (
            (index + 1)
            / horizon
        )

        tracking_cost += (
            progress_weight
            * squared_distance
            / position_normalization
        )

        # ====================================================
        # COSTE ENERGÉTICO
        # ====================================================

        v = (
            controls[
                index,
                0,
            ]
        )

        omega = (
            controls[
                index,
                1,
            ]
        )

        normalized_v = (
            v
            / (
                max_speed
                + epsilon
            )
        )

        normalized_omega = (
            omega
            / (
                max_omega
                + epsilon
            )
        )

        energy_cost += (
            normalized_v ** 2
            + normalized_omega ** 2
        )

        # ====================================================
        # COSTE DE SUAVIDAD
        # ====================================================

        delta_v = (
            v
            - previous_v
        )

        delta_omega = (
            omega
            - previous_omega
        )

        normalized_delta_v = (
            delta_v
            / (
                max_speed
                + epsilon
            )
        )

        normalized_delta_omega = (
            delta_omega
            / (
                max_omega
                + epsilon
            )
        )

        smoothness_cost += (
            normalized_delta_v ** 2
            + normalized_delta_omega ** 2
        )

        previous_v = v
        previous_omega = omega

    # ========================================================
    # COSTE TERMINAL DE POSICIÓN
    # ========================================================

    final_x = (
        predicted_states[
            -1,
            0,
        ]
    )

    final_y = (
        predicted_states[
            -1,
            1,
        ]
    )

    final_reference_x = (
        reference_points[
            -1,
            0,
        ]
    )

    final_reference_y = (
        reference_points[
            -1,
            1,
        ]
    )

    final_dx = (
        final_x
        - final_reference_x
    )

    final_dy = (
        final_y
        - final_reference_y
    )

    final_distance = (
        math.hypot(
            final_dx,
            final_dy,
        )
    )

    terminal_position_cost = (
        final_dx ** 2
        + final_dy ** 2
    ) / position_normalization

    # ========================================================
    # COSTE TERMINAL DE VELOCIDAD
    #
    # Solo se activa al acercarnos al objetivo.
    # ========================================================

    final_v = (
        controls[
            -1,
            0,
        ]
    )

    final_omega = (
        controls[
            -1,
            1,
        ]
    )

    terminal_velocity_cost = (
        (
            final_v
            / (
                max_speed
                + epsilon
            )
        ) ** 2
        +
        (
            final_omega
            / (
                max_omega
                + epsilon
            )
        ) ** 2
    )

    # ==========================================
    # ACTIVACIÓN PROGRESIVA DEL FRENADO
    # ==========================================

    stop_activation = clamp(
        (
            stop_radius
            - final_distance
        )
        / max(
            stop_radius,
            epsilon,
        ),
        0.0,
        1.0,
    )

    terminal_velocity_cost *= (
        stop_activation
    )

    # ========================================================
    # COSTE TOTAL
    # ========================================================

    total_cost = (
        weight_tracking
        * tracking_cost

        + weight_energy
        * energy_cost

        + weight_terminal_position
        * terminal_position_cost

        + weight_terminal_velocity
        * terminal_velocity_cost

        + weight_smoothness
        * smoothness_cost
    )

    return float(
        total_cost
    )


# ============================================================
# RESTRICCIONES DEL CONTROL
# ============================================================


def calculate_rate_constraints(
    flat_controls: np.ndarray,
    previous_control: np.ndarray,
    horizon: int,
    dt: float,
    max_acceleration: float,
    max_angular_acceleration: float,
) -> np.ndarray:
    """
    Impone:

        |v(k)-v(k-1)|
            <= a_max Ts

        |omega(k)-omega(k-1)|
            <= alpha_max Ts
    """

    controls = (
        flat_controls.reshape(
            horizon,
            2,
        )
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

    previous_v = (
        previous_control[0]
    )

    previous_omega = (
        previous_control[1]
    )

    for index in range(
        horizon
    ):

        current_v = (
            controls[
                index,
                0,
            ]
        )

        current_omega = (
            controls[
                index,
                1,
            ]
        )

        delta_v = (
            current_v
            - previous_v
        )

        delta_omega = (
            current_omega
            - previous_omega
        )

        constraints.append(
            maximum_delta_v
            - delta_v
        )

        constraints.append(
            maximum_delta_v
            + delta_v
        )

        constraints.append(
            maximum_delta_omega
            - delta_omega
        )

        constraints.append(
            maximum_delta_omega
            + delta_omega
        )

        previous_v = (
            current_v
        )

        previous_omega = (
            current_omega
        )

    return np.array(
        constraints,
        dtype=float,
    )


# ============================================================
# WARM START
# ============================================================


def build_initial_guess(
    previous_solution: np.ndarray | None,
    previous_control: np.ndarray,
    state: np.ndarray,
    reference_points: np.ndarray,
    horizon: int,
    dt: float,
    max_speed: float,
    max_omega: float,
    max_acceleration: float,
    max_angular_acceleration: float,
) -> np.ndarray:
    """
    Genera la solución inicial de SLSQP.

    Si la solución anterior tiene movimiento:
        utiliza warm start.

    Si está prácticamente parada:
        descarta ese warm start y genera uno nuevo
        orientado hacia el objetivo.
    """

    if previous_solution is not None:

        shifted = np.vstack(
            [
                previous_solution[
                    1:
                ],
                previous_solution[
                    -1
                ],
            ]
        )

        # ==========================================
        # COMPROBAR SI EL WARM START ESTÁ BLOQUEADO
        # ==========================================

        mean_speed = float(
            np.mean(
                np.abs(
                    shifted[
                        :,
                        0,
                    ]
                )
            )
        )

        mean_omega = float(
            np.mean(
                np.abs(
                    shifted[
                        :,
                        1,
                    ]
                )
            )
        )

        # Si todavía hay movimiento suficiente,
        # reutilizamos la solución anterior.
        distance_to_reference = math.hypot(
            reference_points[0, 0] - state[0],
            reference_points[0, 1] - state[1],
        )

        minimum_moving_speed = 5.0
        restart_distance = 60.0

        robot_is_stalled = (
            mean_speed < minimum_moving_speed
            and distance_to_reference > restart_distance
        )

        if not robot_is_stalled:
            return shifted.flatten()

        # Si no, continuamos y generamos
        # una nueva solución inicial.

    # ========================================================
    # NUEVA SOLUCIÓN INICIAL
    # ========================================================

    controls = np.zeros(
        (
            horizon,
            2,
        ),
        dtype=float,
    )

    current_v = float(
        previous_control[0]
    )

    current_omega = float(
        previous_control[1]
    )

    robot_x = (
        state[0]
    )

    robot_y = (
        state[1]
    )

    theta = (
        state[2]
    )

    first_reference_x = (
        reference_points[
            0,
            0,
        ]
    )

    first_reference_y = (
        reference_points[
            0,
            1,
        ]
    )

    dx = (
        first_reference_x
        - robot_x
    )

    dy = (
        first_reference_y
        - robot_y
    )

    desired_heading = (
        math.atan2(
            dy,
            dx,
        )
    )

    heading_error = (
        normalize_angle(
            desired_heading
            - theta
        )
    )

    target_omega = clamp(
        2.0
        * heading_error,
        -max_omega,
        max_omega,
    )

    maximum_delta_v = (
        max_acceleration
        * dt
    )

    maximum_delta_omega = (
        max_angular_acceleration
        * dt
    )

    # Si estamos muy mal orientados,
    # no tiene sentido acelerar al máximo.
    alignment = max(
        0.0,
        math.cos(
            heading_error
        ),
    )

    target_speed = (
        max_speed
        * max(
            0.2,
            alignment,
        )
    )

    for index in range(
        horizon
    ):

        current_v = move_towards(
            current=current_v,
            target=target_speed,
            maximum_change=(
                maximum_delta_v
                * 0.5
            ),
        )

        current_omega = move_towards(
            current=current_omega,
            target=target_omega,
            maximum_change=(
                maximum_delta_omega
                * 0.5
            ),
        )

        controls[
            index,
            0,
        ] = current_v

        controls[
            index,
            1,
        ] = current_omega

    return (
        controls.flatten()
    )


# ============================================================
# RESOLVER MPC
# ============================================================


def solve_mpc(
    state: np.ndarray,
    reference_points: np.ndarray,
    previous_control: np.ndarray,
    previous_solution: np.ndarray | None,
    horizon: int,
    dt: float,
    position_scale: float,
    stop_radius: float,
    max_speed: float,
    max_omega: float,
    max_acceleration: float,
    max_angular_acceleration: float,
    weight_tracking: float,
    weight_energy: float,
    weight_terminal_position: float,
    weight_terminal_velocity: float,
    weight_smoothness: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    bool,
    int,
]:
    """
    Resuelve el MPC utilizando SLSQP.
    """

    initial_guess = (
        build_initial_guess(
            previous_solution=(
                previous_solution
            ),
            previous_control=(
                previous_control
            ),
            state=state,
            reference_points=(
                reference_points
            ),
            horizon=horizon,
            dt=dt,
            max_speed=max_speed,
            max_omega=max_omega,
            max_acceleration=(
                max_acceleration
            ),
            max_angular_acceleration=(
                max_angular_acceleration
            ),
        )
    )

    bounds = []

    for _ in range(
        horizon
    ):

        # v
        bounds.append(
            (
                0.0,
                max_speed,
            )
        )

        # omega
        bounds.append(
            (
                -max_omega,
                max_omega,
            )
        )

    constraints = {
        "type": "ineq",
        "fun": (
            calculate_rate_constraints
        ),
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
            reference_points,
            horizon,
            dt,
            position_scale,
            stop_radius,
            max_speed,
            max_omega,
            weight_tracking,
            weight_energy,
            weight_terminal_position,
            weight_terminal_velocity,
            weight_smoothness,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 80,
            "ftol": 1e-6,
            "disp": False,
        },
    )

    # ========================================================
    # VALIDACIÓN DE LA SOLUCIÓN
    # ========================================================

    candidate_valid = (
        result.x is not None
        and np.all(
            np.isfinite(
                result.x
            )
        )
    )

    candidate_feasible = False

    if candidate_valid:

        candidate_constraints = (
            calculate_rate_constraints(
                flat_controls=(
                    result.x
                ),
                previous_control=(
                    previous_control
                ),
                horizon=horizon,
                dt=dt,
                max_acceleration=(
                    max_acceleration
                ),
                max_angular_acceleration=(
                    max_angular_acceleration
                ),
            )
        )

        candidate_feasible = (
            np.min(
                candidate_constraints
            )
            >= -1e-5
        )

    if (
        candidate_valid
        and candidate_feasible
    ):
        optimal_flat = (
            result.x
        )

    else:
        optimal_flat = (
            initial_guess
        )

    optimal_controls = (
        optimal_flat.reshape(
            horizon,
            2,
        )
    )

    distance_to_reference = math.hypot(
        reference_points[0, 0] - state[0],
        reference_points[0, 1] - state[1],
    )

    first_speed = optimal_controls[0, 0]

    stalled_solution = (
        distance_to_reference > 60.0
        and first_speed < 2.0
    )

    predicted_states = (
        predict_states(
            initial_state=state,
            controls=(
                optimal_controls
            ),
            dt=dt,
        )
    )

    mpc_cost = (
        calculate_mpc_cost(
            flat_controls=(
                optimal_flat
            ),
            initial_state=state,
            reference_points=(
                reference_points
            ),
            horizon=horizon,
            dt=dt,
            position_scale=(
                position_scale
            ),
            stop_radius=(
                stop_radius
            ),
            max_speed=max_speed,
            max_omega=max_omega,
            weight_tracking=(
                weight_tracking
            ),
            weight_energy=(
                weight_energy
            ),
            weight_terminal_position=(
                weight_terminal_position
            ),
            weight_terminal_velocity=(
                weight_terminal_velocity
            ),
            weight_smoothness=(
                weight_smoothness
            ),
        )
    )

    return (
        optimal_controls,
        predicted_states,
        float(
            mpc_cost
        ),
        bool(
            result.success
        ),
        int(
            result.nit
        ),
    )


# ============================================================
# DIBUJO
# ============================================================


def draw_robot(
    frame,
    robot_position: tuple[int, int],
    theta: float,
) -> None:

    robot_x, robot_y = (
        robot_position
    )

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
    predicted_states: np.ndarray | None,
) -> None:

    if (
        predicted_states is None
        or len(
            predicted_states
        ) < 2
    ):
        return

    points = np.array(
        [
            [
                int(
                    state[0]
                ),
                int(
                    state[1]
                ),
            ]
            for state
            in predicted_states
        ],
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


def draw_reference_trajectory(
    frame,
    reference_points: np.ndarray | None,
) -> None:

    if (
        reference_points is None
        or len(
            reference_points
        ) < 2
    ):
        return

    points = np.array(
        reference_points,
        dtype=np.int32,
    ).reshape(
        (-1, 1, 2)
    )

    cv2.polylines(
        frame,
        [points],
        False,
        (0, 165, 255),
        3,
    )

    for point in (
        reference_points
    ):

        cv2.circle(
            frame,
            (
                int(
                    point[0]
                ),
                int(
                    point[1]
                ),
            ),
            5,
            (0, 165, 255),
            -1,
        )


# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================


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
    control_period: float,
    position_scale: float,
    stop_radius: float,
    weight_tracking: float,
    weight_energy: float,
    weight_terminal_position: float,
    weight_terminal_velocity: float,
    weight_smoothness: float,
    smoothing_alpha: float,
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

    video_path = (
        resolve_project_path(
            video_path
        )
    )

    guidance_path = (
        resolve_project_path(
            guidance_path
        )
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"No existe el vídeo: {video_path}"
        )

    # ========================================================
    # GUIADO
    # ========================================================

    guidance = (
        load_guidance(
            guidance_path
        )
    )

    guidance = (
        smooth_guidance(
            guidance=guidance,
            alpha=smoothing_alpha,
        )
    )

    # ========================================================
    # VÍDEO
    # ========================================================

    capture = (
        cv2.VideoCapture(
            str(
                video_path
            )
        )
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

    # ========================================================
    # TEMPORIZACIÓN
    # ========================================================

    dt_video = (
        1.0
        / fps
    )

    control_interval_frames = max(
        1,
        int(
            round(
                control_period
                * fps
            )
        ),
    )

    dt_control = (
        control_interval_frames
        / fps
    )

    horizon_seconds = (
        horizon
        * dt_control
    )

    max_omega = (
        math.radians(
            max_omega_degrees
        )
    )

    max_angular_acceleration = (
        math.radians(
            max_angular_acceleration_degrees
        )
    )

    # ========================================================
    # SALIDA
    # ========================================================

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

    video_writer = (
        cv2.VideoWriter(
            str(
                output_video
            ),
            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),
            fps,
            (
                width,
                height,
            ),
        )
    )

    if not video_writer.isOpened():

        capture.release()

        raise RuntimeError(
            "No se ha podido crear el vídeo."
        )

    # ========================================================
    # ESTADO INICIAL
    # ========================================================

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

    last_guidance_data = None

    last_predicted_states = None
    last_reference_points = None

    last_mpc_cost = 0.0

    last_optimizer_success = True

    last_optimizer_iterations = 0

    next_control_frame = 0

    frame_index = 0

    optimizer_failures = 0
    optimizer_updates = 0

    # ========================================================
    # INFORMACIÓN
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "SIMULACIÓN MPC V5"
    )

    print(
        "=" * 70
    )

    print(
        f"FPS vídeo: {fps:.3f}"
    )

    print(
        f"Ts MPC real: "
        f"{dt_control:.4f} s"
    )

    print(
        f"Horizonte Hp: "
        f"{horizon}"
    )

    print(
        f"Horizonte temporal: "
        f"{horizon_seconds:.3f} s"
    )

    print(
        f"Escala posición: "
        f"{position_scale:.1f} px"
    )

    print(
        f"Radio de frenado: "
        f"{stop_radius:.1f} px"
    )

    print(
        f"EMA alpha: "
        f"{smoothing_alpha}"
    )

    print()

    # ========================================================
    # CSV
    # ========================================================

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        csv_writer = (
            csv.writer(
                csv_file
            )
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
                "distance_to_desired",
                "mpc_cost",
                "optimizer_success",
                "optimizer_iterations",
                "mpc_updated",
                "control_period_seconds",
                "horizon_seconds",
            ]
        )

        # ====================================================
        # BUCLE PRINCIPAL
        # ====================================================

        while True:

            success, frame = (
                capture.read()
            )

            if not success:
                break

            new_guidance_data = (
                guidance.get(
                    frame_index
                )
            )

            if new_guidance_data is not None:

                last_guidance_data = (
                    new_guidance_data
                )

            data = (
                last_guidance_data
            )

            mpc_updated = False

            if data is not None:

                # ============================================
                # ACTUALIZAR MPC
                # ============================================

                if (
                    frame_index
                    >= next_control_frame
                ):

                    reference_points = (
                        build_reference_horizon(
                            guidance=guidance,
                            frame_index=(
                                frame_index
                            ),
                            current_data=(
                                data
                            ),
                            horizon=horizon,
                            control_interval_frames=(
                                control_interval_frames
                            ),
                        )
                    )

                    (
                        optimal_controls,
                        predicted_states,
                        mpc_cost,
                        optimizer_success,
                        optimizer_iterations,
                    ) = solve_mpc(
                        state=state,
                        reference_points=(
                            reference_points
                        ),
                        previous_control=(
                            current_control
                        ),
                        previous_solution=(
                            previous_solution
                        ),
                        horizon=horizon,
                        dt=dt_control,
                        position_scale=(
                            position_scale
                        ),
                        stop_radius=(
                            stop_radius
                        ),
                        max_speed=max_speed,
                        max_omega=max_omega,
                        max_acceleration=(
                            max_acceleration
                        ),
                        max_angular_acceleration=(
                            max_angular_acceleration
                        ),
                        weight_tracking=(
                            weight_tracking
                        ),
                        weight_energy=(
                            weight_energy
                        ),
                        weight_terminal_position=(
                            weight_terminal_position
                        ),
                        weight_terminal_velocity=(
                            weight_terminal_velocity
                        ),
                        weight_smoothness=(
                            weight_smoothness
                        ),
                    )

                    # ========================================
                    # SOLO APLICAMOS EL PRIMER CONTROL
                    # ========================================

                    current_control = (
                        optimal_controls[
                            0
                        ].copy()
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

                    last_mpc_cost = (
                        mpc_cost
                    )

                    last_optimizer_success = (
                        optimizer_success
                    )

                    last_optimizer_iterations = (
                        optimizer_iterations
                    )

                    optimizer_updates += 1

                    if not optimizer_success:

                        optimizer_failures += 1

                    mpc_updated = True

                    next_control_frame = (
                        frame_index
                        + control_interval_frames
                    )

                # ============================================
                # SIMULAR ROBOT FRAME A FRAME
                # ============================================

                state = update_state(
                    state=state,
                    control=(
                        current_control
                    ),
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
                    int(
                        state[0]
                    ),
                    int(
                        state[1]
                    ),
                )

                desired_point = (
                    int(
                        data[
                            "desired_x"
                        ]
                    ),
                    int(
                        data[
                            "desired_y"
                        ]
                    ),
                )

                flock_point = (
                    int(
                        data[
                            "flock_x"
                        ]
                    ),
                    int(
                        data[
                            "flock_y"
                        ]
                    ),
                )

                distance_to_desired = (
                    math.hypot(
                        data[
                            "desired_x"
                        ]
                        - state[0],
                        data[
                            "desired_y"
                        ]
                        - state[1],
                    )
                )

                robot_trajectory.append(
                    robot_point
                )

                # ============================================
                # REBAÑO
                # ============================================

                cv2.circle(
                    frame,
                    flock_point,
                    10,
                    (255, 255, 0),
                    -1,
                )

                # ============================================
                # OBJETIVO
                # ============================================

                cv2.circle(
                    frame,
                    desired_point,
                    14,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "OBJETIVO MPC",
                    (
                        desired_point[0]
                        + 20,
                        desired_point[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                # ============================================
                # REFERENCIA FUTURA
                # ============================================

                draw_reference_trajectory(
                    frame=frame,
                    reference_points=(
                        last_reference_points
                    ),
                )

                # ============================================
                # PREDICCIÓN FUTURA DEL ROBOT
                # ============================================

                draw_predicted_trajectory(
                    frame=frame,
                    predicted_states=(
                        last_predicted_states
                    ),
                )

                # ============================================
                # TRAYECTORIA REAL
                # ============================================

                if len(
                    robot_trajectory
                ) >= 2:

                    trajectory_points = (
                        np.array(
                            robot_trajectory,
                            dtype=np.int32,
                        ).reshape(
                            (-1, 1, 2)
                        )
                    )

                    cv2.polylines(
                        frame,
                        [
                            trajectory_points
                        ],
                        False,
                        (255, 0, 255),
                        3,
                    )

                # ============================================
                # ROBOT
                # ============================================

                draw_robot(
                    frame=frame,
                    robot_position=(
                        robot_point
                    ),
                    theta=state[2],
                )

                theta_degrees = (
                    math.degrees(
                        state[2]
                    )
                )

                omega_degrees = (
                    math.degrees(
                        current_control[
                            1
                        ]
                    )
                )

                # ============================================
                # INFORMACIÓN
                # ============================================

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
                    (
                        f"Ts MPC: "
                        f"{dt_control:.3f} s"
                    ),
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Horizonte: "
                        f"{horizon_seconds:.2f} s"
                    ),
                    (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"v: "
                        f"{current_control[0]:.1f} px/s"
                    ),
                    (30, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"omega: "
                        f"{omega_degrees:.1f} deg/s"
                    ),
                    (30, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Distancia: "
                        f"{distance_to_desired:.1f} px"
                    ),
                    (30, 245),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"Coste MPC: "
                        f"{last_mpc_cost:.5f}"
                    ),
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
                    (
                        f"Optimizador: "
                        f"{optimizer_text}"
                    ),
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
                        f"Stop radius: "
                        f"{stop_radius:.0f} px"
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

                # ============================================
                # CSV
                # ============================================

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{frame_index / fps:.3f}",
                        f"{state[0]:.3f}",
                        f"{state[1]:.3f}",
                        f"{theta_degrees:.3f}",
                        f"{current_control[0]:.3f}",
                        f"{omega_degrees:.3f}",
                        f"{data['desired_x']:.3f}",
                        f"{data['desired_y']:.3f}",
                        f"{distance_to_desired:.3f}",
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

            if (
                frame_index
                % 250
                == 0
            ):

                print(
                    f"{frame_index}/{total_frames}"
                )

    # ========================================================
    # FINAL
    # ========================================================

    capture.release()

    video_writer.release()

    print()

    print(
        "=" * 70
    )

    print(
        "SIMULACIÓN MPC V5 FINALIZADA"
    )

    print(
        "=" * 70
    )

    print(
        f"Frames procesados: "
        f"{frame_index}"
    )

    print(
        f"Actualizaciones MPC: "
        f"{optimizer_updates}"
    )

    print(
        f"Fallos optimizador: "
        f"{optimizer_failures}"
    )

    if optimizer_updates > 0:

        failure_percentage = (
            100.0
            * optimizer_failures
            / optimizer_updates
        )

        print(
            f"Porcentaje fallos: "
            f"{failure_percentage:.2f}%"
        )

    print(
        f"Vídeo: {output_video}"
    )

    print(
        f"CSV: {output_csv}"
    )


# ============================================================
# ARGUMENTOS
# ============================================================


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "MPC v5 para seguimiento "
            "del punto de conducción."
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
        "--control-period",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--position-scale",
        type=float,
        default=200.0,
        help=(
            "Escala de normalización "
            "del error espacial en píxeles."
        ),
    )

    parser.add_argument(
        "--stop-radius",
        type=float,
        default=40.0,
        help=(
            "Distancia a partir de la cual "
            "empieza a penalizarse la velocidad "
            "terminal."
        ),
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
        default=0.2,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="mpc_v5",
    )

    return (
        parser.parse_args()
    )


def main():

    args = (
        parse_arguments()
    )

    simulate_robot_mpc(
        video_path=args.video,
        guidance_path=args.guidance,
        robot_start_x=(
            args.robot_start_x
        ),
        robot_start_y=(
            args.robot_start_y
        ),
        initial_heading_degrees=(
            args.initial_heading
        ),
        max_speed=(
            args.max_speed
        ),
        max_omega_degrees=(
            args.max_omega
        ),
        max_acceleration=(
            args.max_acceleration
        ),
        max_angular_acceleration_degrees=(
            args.max_angular_acceleration
        ),
        horizon=(
            args.horizon
        ),
        control_period=(
            args.control_period
        ),
        position_scale=(
            args.position_scale
        ),
        stop_radius=(
            args.stop_radius
        ),
        weight_tracking=(
            args.weight_tracking
        ),
        weight_energy=(
            args.weight_energy
        ),
        weight_terminal_position=(
            args.weight_terminal_position
        ),
        weight_terminal_velocity=(
            args.weight_terminal_velocity
        ),
        weight_smoothness=(
            args.weight_smoothness
        ),
        smoothing_alpha=(
            args.smoothing_alpha
        ),
        run_name=(
            args.name
        ),
    )


if __name__ == "__main__":
    main()