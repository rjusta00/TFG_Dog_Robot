import argparse
import json
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


def evaluate_split(
    model: YOLO,
    dataset_yaml: Path,
    split_name: str,
    image_size: int,
    batch_size: int,
    device: str | None,
    training_directory: Path,
) -> tuple[Path, dict[str, float]]:
    """Evalua un split y guarda sus curvas y matrices en una carpeta propia."""

    results = model.val(
        data=str(dataset_yaml),
        split=split_name,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        plots=True,
        verbose=True,
        project=str(training_directory / "evaluation"),
        name=split_name,
        exist_ok=True,
    )

    metrics = {
        key: float(value)
        for key, value in results.results_dict.items()
    }

    return Path(results.save_dir).resolve(), metrics


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

    # El entrenamiento ya guarda las curvas por época en results.png y
    # results.csv. Re-evaluamos best.pt para obtener gráficas de detección
    # finales, sin depender del último checkpoint entrenado.
    best_model = YOLO(str(best_weights))
    evaluation_summary: dict[str, dict[str, float | str]] = {}

    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL DE VALIDACIÓN")
    print("=" * 70)

    validation_directory, validation_metrics = evaluate_split(
        model=best_model,
        dataset_yaml=dataset_yaml,
        split_name="val",
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        training_directory=training_directory,
    )
    evaluation_summary["validation"] = {
        "artifacts_directory": str(validation_directory),
        **validation_metrics,
    }
    print(f"Resultados de validación: {validation_directory}")

    # Si el dataset define un split test, ejecutamos una evaluación final
    # totalmente separada de la selección realizada con validación.
    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL")
    print("=" * 70)

    try:
        test_directory, test_metrics = evaluate_split(
            model=best_model,
            dataset_yaml=dataset_yaml,
            split_name="test",
            image_size=image_size,
            batch_size=batch_size,
            device=device,
            training_directory=training_directory,
        )
        evaluation_summary["test"] = {
            "artifacts_directory": str(test_directory),
            **test_metrics,
        }

        print(
            f"Evaluación test completada: {test_directory}"
        )

    except Exception as error:
        print(
            "No se pudo ejecutar la evaluación sobre el split test. "
            "Continuando sin esa evaluación."
        )
        print(f"Motivo: {error}")

    summary_path = training_directory / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(evaluation_summary, indent=2),
        encoding="utf-8",
    )

    print("\nArtefactos para el informe:")
    print(f"  - Curvas de entrenamiento y loss: {training_directory / 'results.png'}")
    print(f"  - Valores por época: {training_directory / 'results.csv'}")
    print(f"  - Matrices y curvas de validación: {validation_directory}")
    print(f"  - Resumen de métricas: {summary_path}")


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
