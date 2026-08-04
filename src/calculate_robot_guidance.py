import argparse
import csv
import math
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "guidance"


def resolve_project_path(path: Path) -> Path:
    """
    Convierte una ruta relativa en absoluta usando como referencia
    la raíz del proyecto.
    """

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def load_flock_trajectory(
    trajectory_path: Path,
) -> dict[int, dict[str, float]]:
    """
    Carga el centro y la bounding box del rebaño para cada frame.
    """

    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el CSV de trayectoria: {trajectory_path}"
        )

    trajectory: dict[int, dict[str, float]] = {}

    with trajectory_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "frame",
            "center_x",
            "center_y",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "confidence",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "El CSV no contiene las columnas necesarias: "
                + ", ".join(sorted(missing_columns))
                + ". Vuelve a ejecutar track_flock_motion.py."
            )

        for row in reader:
            if not row["center_x"] or not row["center_y"]:
                continue

            frame_index = int(row["frame"])

            trajectory[frame_index] = {
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "box_x1": float(row["box_x1"]),
                "box_y1": float(row["box_y1"]),
                "box_x2": float(row["box_x2"]),
                "box_y2": float(row["box_y2"]),
                "confidence": (
                    float(row["confidence"])
                    if row["confidence"]
                    else 0.0
                ),
            }

    if not trajectory:
        raise RuntimeError(
            "El CSV no contiene ninguna posición válida del rebaño."
        )

    return trajectory


def clip_point(
    point: tuple[int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    """
    Impide que un punto quede fuera de la imagen.
    """

    x, y = point

    x = max(0, min(frame_width - 1, x))
    y = max(0, min(frame_height - 1, y))

    return x, y


def calculate_driving_point(
    flock_center: tuple[int, int],
    flock_box: tuple[float, float, float, float],
    target_point: tuple[int, int],
    safety_margin: float,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    float,
    float,
]:
    """
    Calcula una posición del robot situada fuera de la bounding box.

    Pasos:

    1. Obtiene la dirección del centro del rebaño al destino.
    2. Busca el borde posterior de la bounding box.
    3. Añade un margen de seguridad más allá de ese borde.

    Devuelve:
        - posición deseada del robot;
        - punto del borde posterior;
        - distancia del rebaño al destino;
        - distancia desde el centro al borde posterior.
    """

    center_x, center_y = flock_center
    target_x, target_y = target_point
    x1, y1, x2, y2 = flock_box

    vector_x = target_x - center_x
    vector_y = target_y - center_y

    distance_to_target = math.hypot(
        vector_x,
        vector_y,
    )

    if distance_to_target < 1e-6:
        return (
            flock_center,
            flock_center,
            0.0,
            0.0,
        )

    # Dirección desde el rebaño hacia el destino.
    unit_x = vector_x / distance_to_target
    unit_y = vector_y / distance_to_target

    # Dirección contraria: lado posterior del rebaño.
    back_x = -unit_x
    back_y = -unit_y

    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)

    half_width = box_width / 2.0
    half_height = box_height / 2.0

    epsilon = 1e-6

    # Distancia necesaria para alcanzar un borde vertical.
    distance_to_vertical_edge = (
        half_width / abs(back_x)
        if abs(back_x) > epsilon
        else float("inf")
    )

    # Distancia necesaria para alcanzar un borde horizontal.
    distance_to_horizontal_edge = (
        half_height / abs(back_y)
        if abs(back_y) > epsilon
        else float("inf")
    )

    # El primer borde que toca el vector al salir de la caja.
    distance_to_rear_edge = min(
        distance_to_vertical_edge,
        distance_to_horizontal_edge,
    )

    rear_edge_x = (
        center_x
        + back_x * distance_to_rear_edge
    )

    rear_edge_y = (
        center_y
        + back_y * distance_to_rear_edge
    )

    desired_robot_x = (
        rear_edge_x
        + back_x * safety_margin
    )

    desired_robot_y = (
        rear_edge_y
        + back_y * safety_margin
    )

    rear_edge_point = (
        int(round(rear_edge_x)),
        int(round(rear_edge_y)),
    )

    desired_robot_point = (
        int(round(desired_robot_x)),
        int(round(desired_robot_y)),
    )

    return (
        desired_robot_point,
        rear_edge_point,
        distance_to_target,
        distance_to_rear_edge,
    )


def draw_guidance(
    frame,
    flock_center: tuple[int, int],
    flock_box: tuple[float, float, float, float],
    target_point: tuple[int, int],
    rear_edge_point: tuple[int, int],
    driving_point: tuple[int, int],
    distance_to_target: float,
    confidence: float,
    target_reached: bool,
) -> None:
    """
    Dibuja la caja del rebaño, su centro, el borde posterior,
    el destino y la posición deseada del robot.
    """

    x1, y1, x2, y2 = flock_box

    # Bounding box del rebaño.
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255, 255, 0),
        3,
    )

    # Centro del rebaño.
    cv2.circle(
        frame,
        flock_center,
        10,
        (255, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        "CENTRO REBANO",
        (
            flock_center[0] + 15,
            flock_center[1] - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Destino.
    cv2.circle(
        frame,
        target_point,
        20,
        (0, 255, 0),
        4,
    )

    cv2.putText(
        frame,
        "DESTINO",
        (
            target_point[0] + 25,
            target_point[1],
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.arrowedLine(
        frame,
        flock_center,
        target_point,
        (0, 255, 0),
        4,
        tipLength=0.03,
    )

    # Punto donde termina físicamente el rebaño.
    cv2.circle(
        frame,
        rear_edge_point,
        10,
        (0, 165, 255),
        -1,
    )

    cv2.putText(
        frame,
        "BORDE POSTERIOR",
        (
            rear_edge_point[0] + 15,
            rear_edge_point[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    # Posición deseada del robot.
    cv2.circle(
        frame,
        driving_point,
        14,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        "POSICION DESEADA ROBOT",
        (
            driving_point[0] + 20,
            driving_point[1] - 15,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Línea del margen de seguridad.
    cv2.line(
        frame,
        driving_point,
        rear_edge_point,
        (0, 0, 255),
        4,
    )

    # Presión del robot hacia el rebaño.
    cv2.arrowedLine(
        frame,
        driving_point,
        flock_center,
        (0, 0, 255),
        4,
        tipLength=0.08,
    )

    status = (
        "OBJETIVO ALCANZADO"
        if target_reached
        else "CONDUCIENDO"
    )

    cv2.putText(
        frame,
        f"Estado: {status}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Distancia al destino: {distance_to_target:.1f} px",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Confianza flock: {confidence:.2f}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def generate_guidance_video(
    video_path: Path,
    trajectory_path: Path,
    target_x: int,
    target_y: int,
    safety_margin: float,
    goal_radius: float,
    run_name: str,
) -> None:
    """
    Genera un vídeo donde se muestra la posición deseada del robot
    en cada frame.
    """

    video_path = resolve_project_path(video_path)
    trajectory_path = resolve_project_path(trajectory_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el vídeo: {video_path}"
        )

    trajectory = load_flock_trajectory(
        trajectory_path
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"No se ha podido abrir el vídeo: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    frame_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            "No se han podido obtener los FPS del vídeo."
        )

    target_point = clip_point(
        point=(target_x, target_y),
        frame_width=frame_width,
        frame_height=frame_height,
    )

    output_directory = (
        OUTPUT_ROOT
        / run_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video_path = (
        output_directory
        / f"{video_path.stem}_guidance.mp4"
    )

    output_csv_path = (
        output_directory
        / f"{video_path.stem}_guidance.csv"
    )

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se ha podido crear el vídeo: {output_video_path}"
        )

    frame_index = 0
    frames_with_guidance = 0

    print("=" * 70)
    print("CÁLCULO DE LA POSICIÓN DESEADA DEL ROBOT")
    print("=" * 70)
    print(f"Vídeo: {video_path}")
    print(f"Trayectoria: {trajectory_path}")
    print(f"Destino: {target_point}")
    print(
        f"Margen desde el borde posterior: "
        f"{safety_margin} píxeles"
    )
    print(f"Radio del objetivo: {goal_radius} píxeles")
    print()

    with output_csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "frame",
                "time_seconds",
                "flock_center_x",
                "flock_center_y",
                "target_x",
                "target_y",
                "desired_robot_x",
                "desired_robot_y",
                "distance_to_target",
                "status",
                "confidence",
            ]
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            current_time = frame_index / fps

            flock_data = trajectory.get(
                frame_index
            )

            if flock_data is not None:
                center_x = int(round(flock_data["center_x"]))
                center_y = int(round(flock_data["center_y"]))

                confidence = flock_data["confidence"]

                flock_box = (
                    flock_data["box_x1"],
                    flock_data["box_y1"],
                    flock_data["box_x2"],
                    flock_data["box_y2"],
                )

                flock_center = (
                    center_x,
                    center_y,
                )

                (
                    driving_point,
                    rear_edge_point,
                    distance_to_target,
                    distance_to_rear_edge,
                ) = calculate_driving_point(
                    flock_center=flock_center,
                    flock_box=flock_box,
                    target_point=target_point,
                    safety_margin=safety_margin,
                )

                driving_point = clip_point(
                    point=driving_point,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                rear_edge_point = clip_point(
                    point=rear_edge_point,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )

                target_reached = (
                    distance_to_target <= goal_radius
                )

                status = (
                    "TARGET_REACHED"
                    if target_reached
                    else "DRIVING"
                )

                # Una vez alcanzado el destino, el robot debería parar.
                if target_reached:
                    driving_point = flock_center

                draw_guidance(
                    frame=frame,
                    flock_center=flock_center,
                    flock_box=flock_box,
                    target_point=target_point,
                    rear_edge_point=rear_edge_point,
                    driving_point=driving_point,
                    distance_to_target=distance_to_target,
                    confidence=confidence,
                    target_reached=target_reached,
                )

                writer.writerow(
                    [
                        frame_index,
                        f"{current_time:.3f}",
                        center_x,
                        center_y,
                        target_point[0],
                        target_point[1],
                        driving_point[0],
                        driving_point[1],
                        f"{distance_to_target:.3f}",
                        status,
                        f"{confidence:.4f}",
                    ]
                )

                frames_with_guidance += 1

            else:
                cv2.putText(
                    frame,
                    "Rebano no detectado",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                writer.writerow(
                    [
                        frame_index,
                        f"{current_time:.3f}",
                        "",
                        "",
                        target_point[0],
                        target_point[1],
                        "",
                        "",
                        "",
                        "NO_DETECTION",
                        "",
                    ]
                )

            video_writer.write(frame)

            frame_index += 1

            if frame_index % 200 == 0:
                print(
                    f"Procesados {frame_index}/{total_frames} frames"
                )

    capture.release()
    video_writer.release()

    print("\n" + "=" * 70)
    print("GUIADO FINALIZADO")
    print("=" * 70)
    print(f"Frames procesados: {frame_index}")
    print(f"Frames con guiado: {frames_with_guidance}")
    print(f"Vídeo generado: {output_video_path}")
    print(f"Datos generados: {output_csv_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula dónde debe colocarse el robot "
            "para conducir el rebaño hacia un destino."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Ruta del vídeo original.",
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        help="CSV generado por track_flock_motion.py.",
    )

    parser.add_argument(
        "--target-x",
        type=int,
        required=True,
        help="Coordenada X del destino.",
    )

    parser.add_argument(
        "--target-y",
        type=int,
        required=True,
        help="Coordenada Y del destino.",
    )

    parser.add_argument(
        "--safety-margin",
        type=float,
        default=100.0,
        help=(
            "Margen entre el borde posterior del rebaño "
            "y la posición deseada del robot."
        ),
    )

    parser.add_argument(
        "--goal-radius",
        type=float,
        default=80.0,
        help="Radio para considerar alcanzado el destino.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="guidance_v1",
        help="Nombre de la ejecución.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    generate_guidance_video(
        video_path=args.video,
        trajectory_path=args.trajectory,
        target_x=args.target_x,
        target_y=args.target_y,
        safety_margin=args.safety_margin,
        goal_radius=args.goal_radius,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()