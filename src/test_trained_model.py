import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from track_flock_motion import calculate_flock_ellipse, draw_flock_ellipse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "dogRobot_v1_best.pt"
RUNS_DIRECTORY = PROJECT_ROOT / "runs" / "predict"


def resolve_project_path(path: Path) -> Path:
    """
    Convierte una ruta relativa en una ruta absoluta
    tomando como referencia la raíz del proyecto.
    """

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def keep_largest_flock(result, model: YOLO):
    """
    Conserva solo el flock de mayor area y mantiene todos los perros.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return result

    boxes_xyxy = result.boxes.xyxy
    class_ids = result.boxes.cls

    keep_indices = []
    flock_candidates = []

    for index, class_id in enumerate(class_ids.tolist()):
        class_name = model.names[int(class_id)]

        if class_name == "flock":
            x1, y1, x2, y2 = boxes_xyxy[index].tolist()
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            flock_candidates.append(
                (
                    area,
                    index,
                )
            )

        else:
            keep_indices.append(index)

    if flock_candidates:
        _, largest_flock_index = max(
            flock_candidates,
            key=lambda candidate: candidate[0],
        )
        keep_indices.append(largest_flock_index)

    keep_indices.sort()

    result.boxes = result.boxes[keep_indices]

    return result


def draw_largest_flock_ellipse(
    frame,
    result,
    model: YOLO,
    previous_angle: float | None,
) -> float:
    """
    Dibuja una elipse estabilizada sobre el rebaño de mayor área.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return previous_angle if previous_angle is not None else 0.0

    boxes_xyxy = result.boxes.xyxy
    class_ids = result.boxes.cls
    largest_flock = None

    for index, class_id in enumerate(class_ids.tolist()):
        class_name = model.names[int(class_id)]

        if class_name != "flock":
            continue

        x1, y1, x2, y2 = boxes_xyxy[index].tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if largest_flock is None or area > largest_flock[0]:
            largest_flock = (area, x1, y1, x2, y2)

    if largest_flock is None:
        return previous_angle if previous_angle is not None else 0.0

    _, x1, y1, x2, y2 = largest_flock
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))

    ellipse = calculate_flock_ellipse(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        dx=0.0,
        dy=0.0,
    )

    draw_flock_ellipse(
        frame=frame,
        ellipse=ellipse,
        color=(255, 255, 0),
        thickness=4,
    )

    cv2.putText(
        frame,
        f"Shape angle: {ellipse[4]:.1f}",
        (center_x + 15, center_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return ellipse[4]


def process_video(
    video_path: Path,
    model_path: Path,
    confidence: float,
    iou: float,
    image_size: int,
    single_flock: bool,
    draw_ellipse: bool,
    run_name: str,
) -> None:
    """
    Ejecuta el modelo personalizado sobre un vídeo.
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

    print("=" * 70)
    print("PRUEBA DEL MODELO ENTRENADO")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Vídeo: {video_path}")
    print(f"Confianza mínima: {confidence}")
    print(f"IoU NMS: {iou}")
    print(f"Resolución: {image_size}")
    print(f"Filtro single flock: {single_flock}")
    print(f"Dibujar elipse: {draw_ellipse}")

    model = YOLO(str(model_path))

    output_directory = RUNS_DIRECTORY / run_name
    output_video_suffix = (
        "largest_flock_ellipse"
        if single_flock and draw_ellipse
        else "ellipse"
        if draw_ellipse
        else "largest_flock"
    )
    output_video_path = output_directory / f"{video_path.stem}_{output_video_suffix}.mp4"
    video_writer = None

    if single_flock or draw_ellipse:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        video_capture = cv2.VideoCapture(str(video_path))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()

        if fps <= 0:
            fps = 30.0

    results = model.predict(
        source=str(video_path),
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        save=not (single_flock or draw_ellipse),
        project=str(RUNS_DIRECTORY),
        name=run_name,
        exist_ok=True,
        stream=True,
        verbose=True,
    )

    processed_frames = 0
    total_detections = 0

    frames_with_flock = 0
    frames_with_dog = 0

    flock_detections = 0
    dog_detections = 0

    flock_confidences = []
    dog_confidences = []

    max_flocks_in_frame = 0
    max_dogs_in_frame = 0
    previous_ellipse_angle: float | None = None


    for result in results:
        processed_frames += 1

        if single_flock:
            result = keep_largest_flock(
                result=result,
                model=model,
            )

        if single_flock or draw_ellipse:
            annotated_frame = result.plot()

            if draw_ellipse:
                previous_ellipse_angle = draw_largest_flock_ellipse(
                    frame=annotated_frame,
                    result=result,
                    model=model,
                    previous_angle=previous_ellipse_angle,
                )

            if video_writer is None:
                frame_height, frame_width = annotated_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame_width, frame_height),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        f"No se ha podido crear el vídeo: {output_video_path}"
                    )

            video_writer.write(annotated_frame)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        frame_flocks = 0
        frame_dogs = 0

        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()

        for class_id, confidence in zip(class_ids, confidences):
            class_name = model.names[int(class_id)]

            total_detections += 1

            if class_name == "flock":
                frame_flocks += 1
                flock_detections += 1
                flock_confidences.append(confidence)

            elif class_name == "dog":
                frame_dogs += 1
                dog_detections += 1
                dog_confidences.append(confidence)

        if frame_flocks > 0:
            frames_with_flock += 1

        if frame_dogs > 0:
            frames_with_dog += 1

        max_flocks_in_frame = max(
            max_flocks_in_frame,
            frame_flocks,
        )

        max_dogs_in_frame = max(
            max_dogs_in_frame,
            frame_dogs,
        )


    average_flock_confidence = (
        sum(flock_confidences) / len(flock_confidences)
        if flock_confidences
        else 0
    )

    average_dog_confidence = (
        sum(dog_confidences) / len(dog_confidences)
        if dog_confidences
        else 0
    )

    if video_writer is not None:
        video_writer.release()


    print("\n" + "=" * 70)
    print("PRUEBA FINALIZADA")
    print("=" * 70)

    print(f"Frames procesados: {processed_frames}")
    print(f"Detecciones totales: {total_detections}")

    print("\nREBAÑO")
    print(f"Detecciones acumuladas: {flock_detections}")
    print(f"Frames con rebaño: {frames_with_flock}")
    print(f"Máximo de rebaños en un frame: {max_flocks_in_frame}")
    print(f"Confianza media: {average_flock_confidence:.4f}")

    print("\nPERRO")
    print(f"Detecciones acumuladas: {dog_detections}")
    print(f"Frames con perro: {frames_with_dog}")
    print(f"Máximo de perros en un frame: {max_dogs_in_frame}")
    print(f"Confianza media: {average_dog_confidence:.4f}")

    print(f"\nResultado guardado en: {output_directory.resolve()}")

    if single_flock or draw_ellipse:
        print(f"Vídeo anotado: {output_video_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prueba un modelo YOLO personalizado."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Ruta del vídeo que se procesará.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo best.pt.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
        help="Confianza mínima. Por defecto: 0.15.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="Umbral IoU para NMS. Por defecto: 0.5.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Resolución de inferencia.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="dogRobot_smoke",
        help="Nombre de la carpeta de resultados.",
    )

    parser.add_argument(
        "--single-flock",
        action="store_true",
        help="Conserva solo el rebaño de mayor área en cada frame.",
    )

    parser.add_argument(
        "--draw-ellipse",
        action="store_true",
        help="Dibuja la elipse estabilizada del rebaño de mayor área.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    process_video(
        video_path=args.video,
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        single_flock=args.single_flock,
        draw_ellipse=args.draw_ellipse,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
