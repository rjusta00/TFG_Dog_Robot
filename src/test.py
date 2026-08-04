import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"

def process_video(
    video_path: Path,
    confidence: float,
    image_size: int,
) -> None:
    """
    Ejecuta un modelo YOLO preentrenado sobre un vídeo.

    Args:
        video_path: Ruta del vídeo que se quiere procesar.
        confidence: Confianza mínima necesaria para aceptar una detección.
        image_size: Resolución utilizada por YOLO durante la inferencia.
    """

    if not video_path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el vídeo: {video_path.resolve()}"
        )

    print(f"Procesando vídeo: {video_path.resolve()}")

    # Carga el modelo pequeño preentrenado de YOLO26.
    # La primera vez descargará automáticamente el archivo yolo26n.pt.
    model = YOLO("yolo26n.pt")

    # Ejecuta la detección sobre todos los frames del vídeo.
    model.predict(
        source=str(video_path),
        conf=confidence,
        imgsz=image_size,
        save=True,
        project=str(RUNS_DIR),
        name="baseline",
        exist_ok=True,
    )

    output_directory = RUNS_DIR / "baseline"

    print("\nProcesamiento completado.")
    print(f"Resultado guardado en: {output_directory.resolve()}")


def parse_arguments() -> argparse.Namespace:
    """
    Define los argumentos que puede recibir el programa desde la terminal.
    """

    parser = argparse.ArgumentParser(
        description="Prueba un modelo YOLO preentrenado sobre un vídeo."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Ruta del vídeo que se quiere procesar.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confianza mínima para aceptar detecciones. Por defecto: 0.25.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Resolución utilizada por YOLO. Por defecto: 960.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    process_video(
        video_path=args.video,
        confidence=args.confidence,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()