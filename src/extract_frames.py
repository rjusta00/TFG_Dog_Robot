import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_directory: Path,
    every_seconds: float,
    max_frames: int | None,
) -> None:
    """
    Extrae fotogramas de un vídeo a intervalos regulares.

    Args:
        video_path:
            Ruta del vídeo de entrada.

        output_directory:
            Carpeta donde se guardarán las imágenes.

        every_seconds:
            Número de segundos que deben pasar entre una imagen y la siguiente.

        max_frames:
            Número máximo de imágenes que se extraerán.
            Si vale None, no se establece ningún límite.
    """

    if not video_path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado el vídeo: {video_path.resolve()}"
        )

    if every_seconds <= 0:
        raise ValueError("--every-seconds debe ser mayor que cero.")

    # Abrimos el archivo de vídeo con OpenCV.
    video_capture = cv2.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        raise RuntimeError(
            f"OpenCV no ha podido abrir el vídeo: {video_path.resolve()}"
        )

    # Obtenemos información del vídeo.
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(
        video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        video_capture.release()
        raise RuntimeError(
            "No se ha podido obtener la velocidad FPS del vídeo."
        )

    video_duration = total_video_frames / fps

    # Calculamos cuántos frames hay que saltar entre imágenes.
    frame_step = max(1, round(fps * every_seconds))

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_frame_index = 0
    saved_images = 0

    print(f"Vídeo: {video_path.resolve()}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames totales: {total_video_frames}")
    print(f"Duración aproximada: {video_duration:.2f} segundos")
    print(f"Se guardará una imagen cada {every_seconds} segundos")
    print(f"Equivale a guardar una imagen cada {frame_step} frames\n")

    while True:
        success, frame = video_capture.read()

        # Cuando no quedan más frames, finaliza el bucle.
        if not success:
            break

        # Solo guardamos los frames que coincidan con el intervalo.
        if current_frame_index % frame_step == 0:
            time_in_seconds = current_frame_index / fps

            image_name = (
                f"{video_path.stem}"
                f"_frame_{current_frame_index:08d}"
                f"_time_{time_in_seconds:010.2f}.jpg"
            )

            image_path = output_directory / image_name

            image_saved = cv2.imwrite(
                str(image_path),
                frame,
            )

            if not image_saved:
                video_capture.release()
                raise RuntimeError(
                    f"No se ha podido guardar la imagen: {image_path}"
                )

            saved_images += 1

            print(
                f"Imagen {saved_images:04d}: "
                f"frame {current_frame_index}, "
                f"segundo {time_in_seconds:.2f}"
            )

            if max_frames is not None and saved_images >= max_frames:
                break

        current_frame_index += 1

    video_capture.release()

    print("\nExtracción finalizada.")
    print(f"Imágenes guardadas: {saved_images}")
    print(f"Carpeta: {output_directory.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae fotogramas de un vídeo para crear un dataset."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Ruta del vídeo de entrada.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta donde se guardarán las imágenes.",
    )

    parser.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="Intervalo en segundos entre imágenes. Por defecto: 1.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cantidad máxima de imágenes que se extraerán.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    extract_frames(
        video_path=args.video,
        output_directory=args.output,
        every_seconds=args.every_seconds,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()