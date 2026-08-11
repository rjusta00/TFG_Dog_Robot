import argparse
import csv
import math
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "robot_control"


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def classify_direction(
    dx: float,
    dy: float,
    dead_zone: float = 5.0,
) -> str:
    """
    Convierte el vector robot -> posición deseada
    en una de las ocho direcciones.
    """

    distance = math.hypot(dx, dy)

    if distance <= dead_zone:
        return "STOP"

    # En OpenCV Y aumenta hacia abajo.
    angle = math.degrees(
        math.atan2(-dy, dx)
    )

    angle = (angle + 360) % 360

    if angle < 22.5 or angle >= 337.5:
        return "ESTE"

    if angle < 67.5:
        return "NORESTE"

    if angle < 112.5:
        return "NORTE"

    if angle < 157.5:
        return "NOROESTE"

    if angle < 202.5:
        return "OESTE"

    if angle < 247.5:
        return "SUROESTE"

    if angle < 292.5:
        return "SUR"

    return "SURESTE"


def move_robot_towards_target(
    current_position: tuple[float, float],
    desired_position: tuple[float, float],
    max_step: float,
) -> tuple[
    tuple[float, float],
    float,
    float,
    float,
]:
    """
    Mueve el robot hacia el punto deseado.

    El robot puede avanzar como máximo max_step píxeles
    en cada frame.
    """

    robot_x, robot_y = current_position
    target_x, target_y = desired_position

    dx = target_x - robot_x
    dy = target_y - robot_y

    distance = math.hypot(dx, dy)

    if distance < 1e-6:
        return (
            current_position,
            dx,
            dy,
            distance,
        )

    step = min(
        max_step,
        distance,
    )

    unit_x = dx / distance
    unit_y = dy / distance

    new_robot_x = robot_x + unit_x * step
    new_robot_y = robot_y + unit_y * step

    return (
        (new_robot_x, new_robot_y),
        dx,
        dy,
        distance,
    )


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
                "flock_x": int(
                    float(row["flock_center_x"])
                ),
                "flock_y": int(
                    float(row["flock_center_y"])
                ),
                "target_x": int(
                    float(row["target_x"])
                ),
                "target_y": int(
                    float(row["target_y"])
                ),
                "desired_x": int(
                    float(row["desired_robot_x"])
                ),
                "desired_y": int(
                    float(row["desired_robot_y"])
                ),
                "status": row["status"],
            }

    return guidance


def simulate_robot(
    video_path: Path,
    guidance_path: Path,
    robot_start_x: float,
    robot_start_y: float,
    speed_pixels: float,
    arrival_radius: float,
    run_name: str,
) -> None:
    """
    Simula un robot intentando seguir el punto de conducción
    calculado para cada frame.
    """

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
            f"No se puede abrir el vídeo: {video_path}"
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

    output_directory = (
        OUTPUT_ROOT / run_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_directory
        / f"{video_path.stem}_robot_control.mp4"
    )

    output_csv = (
        output_directory
        / f"{video_path.stem}_robot_control.csv"
    )

    writer_video = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    robot_position = (
        robot_start_x,
        robot_start_y,
    )

    # Guardamos trayectoria del robot.
    robot_trajectory = []

    frame_index = 0

    print("=" * 70)
    print("SIMULACIÓN DE CONTROL DEL ROBOT")
    print("=" * 70)
    print(f"Robot inicial: {robot_position}")
    print(
        f"Velocidad máxima: "
        f"{speed_pixels} px/frame"
    )
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
                "desired_x",
                "desired_y",
                "dx",
                "dy",
                "distance",
                "command",
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

                desired_position = (
                    data["desired_x"],
                    data["desired_y"],
                )

                current_robot_x = (
                    robot_position[0]
                )

                current_robot_y = (
                    robot_position[1]
                )

                dx = (
                    desired_position[0]
                    - current_robot_x
                )

                dy = (
                    desired_position[1]
                    - current_robot_y
                )

                distance = math.hypot(
                    dx,
                    dy,
                )

                if distance <= arrival_radius:
                    command = "STOP"

                else:
                    command = classify_direction(
                        dx,
                        dy,
                    )

                    (
                        robot_position,
                        _,
                        _,
                        _,
                    ) = move_robot_towards_target(
                        current_position=robot_position,
                        desired_position=desired_position,
                        max_step=speed_pixels,
                    )

                robot_trajectory.append(
                    (
                        int(robot_position[0]),
                        int(robot_position[1]),
                    )
                )

                # Centro del rebaño
                flock_point = (
                    data["flock_x"],
                    data["flock_y"],
                )

                cv2.circle(
                    frame,
                    flock_point,
                    10,
                    (255, 255, 0),
                    -1,
                )

                # Punto deseado
                cv2.circle(
                    frame,
                    desired_position,
                    13,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "OBJETIVO ROBOT",
                    (
                        desired_position[0] + 15,
                        desired_position[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Robot simulado
                current_robot_point = (
                    int(robot_position[0]),
                    int(robot_position[1]),
                )

                cv2.circle(
                    frame,
                    current_robot_point,
                    16,
                    (255, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "ROBOT",
                    (
                        current_robot_point[0] + 20,
                        current_robot_point[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2,
                )

                # Flecha robot -> posición deseada
                cv2.arrowedLine(
                    frame,
                    current_robot_point,
                    desired_position,
                    (255, 0, 255),
                    4,
                    tipLength=0.08,
                )

                # Trayectoria acumulada
                if len(robot_trajectory) >= 2:

                    for index in range(
                        1,
                        len(robot_trajectory),
                    ):
                        cv2.line(
                            frame,
                            robot_trajectory[index - 1],
                            robot_trajectory[index],
                            (255, 0, 255),
                            2,
                        )

                cv2.putText(
                    frame,
                    f"ORDEN ROBOT: {command}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

                cv2.putText(
                    frame,
                    f"Distancia objetivo: {distance:.1f} px",
                    (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{frame_index / fps:.3f}",
                        f"{robot_position[0]:.2f}",
                        f"{robot_position[1]:.2f}",
                        desired_position[0],
                        desired_position[1],
                        f"{dx:.2f}",
                        f"{dy:.2f}",
                        f"{distance:.2f}",
                        command,
                    ]
                )

            video_writer_frame = frame

            writer_video.write(
                video_writer_frame
            )

            frame_index += 1

            if frame_index % 200 == 0:
                print(
                    f"{frame_index}/{total_frames}"
                )

    capture.release()
    writer_video.release()

    print("\nSIMULACIÓN TERMINADA")
    print(f"Vídeo: {output_video}")
    print(f"CSV: {output_csv}")


def parse_arguments():
    parser = argparse.ArgumentParser()

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
        "--speed-pixels",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--arrival-radius",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="robot_control_v1",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    simulate_robot(
        video_path=args.video,
        guidance_path=args.guidance,
        robot_start_x=args.robot_start_x,
        robot_start_y=args.robot_start_y,
        speed_pixels=args.speed_pixels,
        arrival_radius=args.arrival_radius,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()