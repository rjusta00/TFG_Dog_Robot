import argparse
import csv
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "dogRobot_v1_best.pt"
)

OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tracking"


def resolve_project_path(path: Path) -> Path:
    """
    Convierte una ruta relativa en una ruta absoluta tomando
    como referencia la raíz del proyecto.
    """

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def calculate_box_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[int, int]:
    """
    Calcula el centro de una bounding box.
    """

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return center_x, center_y


def classify_direction(
    dx: float,
    dy: float,
    dead_zone: float,
) -> str:
    """
    Convierte un desplazamiento en una de las ocho direcciones.

    En las imágenes:
    - X aumenta hacia la derecha.
    - Y aumenta hacia abajo.

    Por eso usamos -dy al calcular el ángulo.
    """

    distance = math.hypot(dx, dy)

    if distance < dead_zone:
        return "QUIETO"

    angle = math.degrees(
        math.atan2(-dy, dx)
    )

    # Convertimos el ángulo al intervalo 0-360.
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


def calculate_movement(
    trajectory: deque[tuple[int, int]],
    direction_window: int,
    dead_zone: float,
) -> tuple[float, float, str]:
    """
    Calcula el movimiento utilizando varios frames.

    No compara únicamente dos frames consecutivos porque las cajas
    de YOLO pueden temblar ligeramente aunque el rebaño esté quieto.
    """

    if len(trajectory) < 2:
        return 0.0, 0.0, "SIN_DATOS"

    window = min(
        direction_window,
        len(trajectory) - 1,
    )

    previous_x, previous_y = trajectory[-window - 1]
    current_x, current_y = trajectory[-1]

    dx = current_x - previous_x
    dy = current_y - previous_y

    direction = classify_direction(
        dx=dx,
        dy=dy,
        dead_zone=dead_zone,
    )

    return dx, dy, direction


def select_main_flock(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    track_ids: np.ndarray | None,
    active_track_id: int | None,
) -> dict | None:
    """
    Selecciona el rebaño principal.

    Si el tracker mantiene el ID anterior, seguimos ese mismo ID.
    Si no, elegimos la caja de mayor superficie.
    """

    candidates = []

    for index, box in enumerate(boxes_xyxy):
        class_id = int(class_ids[index])

        # flock es la clase 0.
        if class_id != 0:
            continue

        x1, y1, x2, y2 = box.tolist()

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        track_id = None

        if track_ids is not None:
            track_id = int(track_ids[index])

        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(confidences[index]),
                "track_id": track_id,
                "area": area,
            }
        )

    if not candidates:
        return None

    # Intentamos mantener el mismo identificador.
    if active_track_id is not None:
        for candidate in candidates:
            if candidate["track_id"] == active_track_id:
                return candidate

    # Si se perdió el ID, seleccionamos la caja más grande.
    return max(
        candidates,
        key=lambda candidate: candidate["area"],
    )


def draw_trajectory(
    frame: np.ndarray,
    trajectory: deque[tuple[int, int]],
) -> None:
    """
    Dibuja la trayectoria histórica del centro del rebaño.
    """

    if len(trajectory) < 2:
        return

    points = np.array(
        trajectory,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        frame,
        [points],
        isClosed=False,
        color=(255, 255, 0),
        thickness=3,
    )


def track_flock(
    video_path: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
    run_name: str,
    direction_window: int,
    dead_zone: float,
) -> None:
    """
    Detecta y sigue el rebaño en un vídeo.

    Genera:
    - Un vídeo con la caja, centro y trayectoria.
    - Un CSV con posiciones y direcciones por frame.
    """

    video_path = resolve_project_path(video_path)
    model_path = resolve_project_path(model_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el vídeo: {video_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo: {model_path}"
        )

    output_directory = OUTPUT_ROOT / run_name

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video_path = (
        output_directory
        / f"{video_path.stem}_tracked.mp4"
    )

    output_csv_path = (
        output_directory
        / f"{video_path.stem}_trajectory.csv"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"No se ha podido abrir el vídeo: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

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

    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se ha podido crear el vídeo: "
            f"{output_video_path}"
        )

    model = YOLO(
        str(model_path)
    )

    # Guardamos los centros de los últimos frames.
    trajectory: deque[tuple[int, int]] = deque(
        maxlen=150,
    )

    active_track_id: int | None = None

    frame_index = 0
    frames_with_flock = 0

    print("=" * 70)
    print("SEGUIMIENTO DEL REBAÑO")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Vídeo: {video_path}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Salida: {output_video_path}")
    print()

    with output_csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "frame",
                "time_seconds",
                "track_id",
                "center_x",
                "center_y",
                "dx",
                "dy",
                "direction",
                "confidence",
            ]
        )

        while True:
            success, frame = capture.read()

            if not success:
                break

            annotated_frame = frame.copy()

            # persist=True mantiene los identificadores entre frames.
            result = model.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=confidence,
                iou=0.5,
                imgsz=image_size,
                verbose=False,
            )[0]

            selected_flock = None

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):
                boxes_xyxy = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                class_ids = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                track_ids = None

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                selected_flock = select_main_flock(
                    boxes_xyxy=boxes_xyxy,
                    class_ids=class_ids,
                    confidences=confidences,
                    track_ids=track_ids,
                    active_track_id=active_track_id,
                )

            current_time = frame_index / fps

            if selected_flock is not None:
                frames_with_flock += 1

                x1, y1, x2, y2 = selected_flock["box"]

                center_x, center_y = calculate_box_center(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )

                active_track_id = selected_flock["track_id"]

                trajectory.append(
                    (center_x, center_y)
                )

                dx, dy, direction = calculate_movement(
                    trajectory=trajectory,
                    direction_window=direction_window,
                    dead_zone=dead_zone,
                )

                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    3,
                )

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    8,
                    (0, 0, 255),
                    -1,
                )

                draw_trajectory(
                    frame=annotated_frame,
                    trajectory=trajectory,
                )

                # Dibujamos una flecha con el desplazamiento medido.
                if len(trajectory) > direction_window:
                    previous_point = trajectory[
                        -direction_window - 1
                    ]

                    cv2.arrowedLine(
                        annotated_frame,
                        previous_point,
                        (center_x, center_y),
                        (255, 0, 255),
                        4,
                        tipLength=0.25,
                    )

                track_text = (
                    str(active_track_id)
                    if active_track_id is not None
                    else "sin_id"
                )

                label = (
                    f"flock ID:{track_text} "
                    f"conf:{selected_flock['confidence']:.2f}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), max(30, int(y1) - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated_frame,
                    f"Direccion: {direction}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated_frame,
                    f"Movimiento: dx={dx:.0f}, dy={dy:.0f}",
                    (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.writerow(
                    [
                        frame_index,
                        f"{current_time:.3f}",
                        active_track_id,
                        center_x,
                        center_y,
                        f"{dx:.3f}",
                        f"{dy:.3f}",
                        direction,
                        f"{selected_flock['confidence']:.4f}",
                    ]
                )

            else:
                cv2.putText(
                    annotated_frame,
                    "Rebano no detectado",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
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
                        "",
                        "",
                        "",
                        "NO_DETECTION",
                        "",
                    ]
                )

            video_writer.write(
                annotated_frame
            )

            frame_index += 1

            if frame_index % 100 == 0:
                print(
                    f"Procesados {frame_index}/{total_frames} frames"
                )

    capture.release()
    video_writer.release()

    detection_percentage = (
        frames_with_flock / frame_index * 100
        if frame_index > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("SEGUIMIENTO FINALIZADO")
    print("=" * 70)
    print(f"Frames procesados: {frame_index}")
    print(f"Frames con rebaño: {frames_with_flock}")
    print(
        f"Porcentaje con detección: "
        f"{detection_percentage:.2f}%"
    )
    print(f"Vídeo generado: {output_video_path}")
    print(f"Trayectoria CSV: {output_csv_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sigue el centro del rebaño y calcula "
            "su dirección de movimiento."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Ruta del vídeo.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo personalizado.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confianza mínima de detección.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Resolución utilizada por YOLO.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="flock_motion",
        help="Nombre de la ejecución.",
    )

    parser.add_argument(
        "--direction-window",
        type=int,
        default=15,
        help=(
            "Número de frames utilizados para calcular "
            "la dirección."
        ),
    )

    parser.add_argument(
        "--dead-zone",
        type=float,
        default=10.0,
        help=(
            "Movimiento mínimo en píxeles para considerar "
            "que el rebaño se está desplazando."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    track_flock(
        video_path=args.video,
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        run_name=args.name,
        direction_window=args.direction_window,
        dead_zone=args.dead_zone,
    )


if __name__ == "__main__":
    main()