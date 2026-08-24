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
    patience: int,
    device: str | None,
    optimizer: str,
    seed: int,
    cache: bool,
    use_amp: bool,
    close_mosaic: int,
    dropout: float,
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
    print(f"Patience: {patience}")
    print(f"Optimizer: {optimizer}")
    print(f"AMP: {use_amp}")
    print(f"Cache: {cache}")
    print(f"Close mosaic: {close_mosaic}")

    if device:
        print(f"Device: {device}")

    model = YOLO(model_name)

    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,

        # workers=0 evita algunos problemas
        # de multiprocessing en Windows.
        "workers": 0,

        "project": str(
            RUNS_DIRECTORY / "train"
        ),
        "name": run_name,
        "exist_ok": True,

        # Entrenamiento más estable para un dataset pequeño
        # y con objetos difíciles como el perro.
        "patience": patience,
        "optimizer": optimizer,
        "seed": seed,
        "deterministic": True,
        "cache": cache,
        "amp": use_amp,
        "cos_lr": True,
        "close_mosaic": close_mosaic,
        "dropout": dropout,
        "plots": True,
        "verbose": True,
    }

    if device:
        train_kwargs["device"] = device

    results = model.train(
        **train_kwargs,
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

    # Si el dataset define un split test, ejecutamos una
    # evaluación final para no depender solo de validación.
    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL")
    print("=" * 70)

    try:
        test_results = model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=image_size,
            batch=batch_size,
            device=device,
            workers=0,
            plots=True,
            verbose=True,
        )

        print(
            f"Evaluación test completada: {test_results.save_dir}"
        )

    except Exception as error:
        print(
            "No se pudo ejecutar la evaluación sobre el split test. "
            "Continuando sin esa evaluación."
        )
        print(f"Motivo: {error}")


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
        default="yolo26s.pt",
        help="Modelo inicial. Por defecto: yolo26s.pt.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas. Por defecto: 100.",
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
        default=8,
        help="Imágenes por lote.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Épocas sin mejora antes de parar.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo, por ejemplo cpu, 0 o 0,1.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizador de Ultralytics. Por defecto: auto.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Carga imágenes en caché para acelerar el entrenamiento.",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Desactiva mixed precision si diera problemas.",
    )

    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Cierra mosaic en las últimas épocas para mejorar cajas finales.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout del head. Por defecto: 0.0.",
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
        patience=args.patience,
        device=args.device,
        optimizer=args.optimizer,
        seed=args.seed,
        cache=args.cache,
        use_amp=not args.no_amp,
        close_mosaic=args.close_mosaic,
        dropout=args.dropout,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
