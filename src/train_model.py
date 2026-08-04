import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = PROJECT_ROOT / "runs"
MODELS_DIRECTORY = PROJECT_ROOT / "models"


def resolve_project_path(path: Path) -> Path:
    """
    Convierte rutas relativas en rutas absolutas
    tomando como base la raíz del proyecto.
    """

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def train_model(
    dataset_yaml: Path,
    model_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    run_name: str,
) -> None:
    """
    Entrena un modelo YOLO con un dataset personalizado.
    """

    dataset_yaml = resolve_project_path(
        dataset_yaml
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"No existe el archivo del dataset: "
            f"{dataset_yaml}"
        )

    print("=" * 70)
    print("ENTRENAMIENTO YOLO")
    print("=" * 70)
    print(f"Dataset: {dataset_yaml}")
    print(f"Modelo inicial: {model_name}")
    print(f"Épocas: {epochs}")
    print(f"Resolución: {image_size}")
    print(f"Batch: {batch_size}")

    model = YOLO(model_name)

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,

        # workers=0 evita algunos problemas
        # de multiprocessing en Windows.
        workers=0,

        project=str(
            RUNS_DIRECTORY / "train"
        ),
        name=run_name,
        exist_ok=True,

        patience=10,
        plots=True,
        verbose=True,
    )

    training_directory = Path(
        results.save_dir
    ).resolve()

    best_weights = (
        training_directory
        / "weights"
        / "best.pt"
    )

    last_weights = (
        training_directory
        / "weights"
        / "last.pt"
    )

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 70)
    print(f"Resultados: {training_directory}")
    print(f"Últimos pesos: {last_weights}")
    print(f"Mejores pesos: {best_weights}")

    if not best_weights.exists():
        raise RuntimeError(
            f"No se ha generado best.pt en: "
            f"{best_weights}"
        )

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_model = (
        MODELS_DIRECTORY
        / f"{run_name}_best.pt"
    )

    shutil.copy2(
        best_weights,
        destination_model,
    )

    print(
        f"Modelo copiado a: "
        f"{destination_model.resolve()}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un detector YOLO personalizado."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Ruta al archivo dataset.yaml.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Modelo inicial. Por defecto: yolo26n.pt.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número de épocas. Por defecto: 20.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Resolución de entrenamiento.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Imágenes por lote.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="dogRobot_smoke",
        help="Nombre de la ejecución.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    train_model(
        dataset_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()