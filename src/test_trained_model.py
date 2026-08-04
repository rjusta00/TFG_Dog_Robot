import argparse
from pathlib import Path

from ultralytics import YOLO


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


def process_video(
    video_path: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
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
    print(f"Resolución: {image_size}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(video_path),
        conf=confidence,
        iou=0.5,
        imgsz=image_size,
        save=True,
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


    for result in results:
        processed_frames += 1

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

    if model.predictor is not None:
        output_directory = Path(model.predictor.save_dir)
    else:
        output_directory = RUNS_DIRECTORY / run_name

    print(f"\nResultado guardado en: {output_directory.resolve()}")


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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    process_video(
        video_path=args.video,
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()