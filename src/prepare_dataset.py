import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_images(images_directory: Path) -> list[Path]:
    """
    Busca todas las imágenes dentro de una carpeta y sus subcarpetas.
    """

    return sorted(
        path
        for path in images_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label(
    labels_directory: Path,
    image_path: Path,
) -> Path | None:
    """
    Busca el archivo TXT que tenga el mismo nombre que una imagen.
    """

    matching_labels = list(
        labels_directory.rglob(f"{image_path.stem}.txt")
    )

    if not matching_labels:
        return None

    if len(matching_labels) > 1:
        raise RuntimeError(
            f"Hay varias etiquetas para la imagen {image_path.name}: "
            f"{matching_labels}"
        )

    return matching_labels[0]


def copy_example(
    image_path: Path,
    label_path: Path,
    output_directory: Path,
    split_name: str,
) -> None:
    """
    Copia una imagen y su etiqueta al split correspondiente.
    """

    images_output = (
        output_directory
        / "images"
        / split_name
    )

    labels_output = (
        output_directory
        / "labels"
        / split_name
    )

    images_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        image_path,
        images_output / image_path.name,
    )

    shutil.copy2(
        label_path,
        labels_output / label_path.name,
    )


def create_dataset_yaml(
    output_directory: Path,
) -> Path:
    """
    Crea el archivo dataset.yaml utilizado por Ultralytics.
    """

    yaml_path = output_directory / "dataset.yaml"

    dataset_root = output_directory.resolve().as_posix()

    yaml_content = f"""path: {dataset_root}

train: images/train
val: images/val

names:
  0: flock
  1: dog
"""

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    return yaml_path


def prepare_dataset(
    images_directory: Path,
    labels_directory: Path,
    output_directory: Path,
    train_ratio: float,
    seed: int,
) -> None:
    """
    Divide un conjunto de imágenes etiquetadas en train y val.
    """

    if not images_directory.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de imágenes: "
            f"{images_directory.resolve()}"
        )

    if not labels_directory.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de etiquetas: "
            f"{labels_directory.resolve()}"
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio debe ser mayor que 0 y menor que 1."
        )

    images = find_images(images_directory)

    if not images:
        raise RuntimeError(
            f"No se encontraron imágenes en: "
            f"{images_directory.resolve()}"
        )

    examples: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []

    for image_path in images:
        label_path = find_label(
            labels_directory=labels_directory,
            image_path=image_path,
        )

        if label_path is None:
            missing_labels.append(image_path)
            continue

        examples.append(
            (
                image_path,
                label_path,
            )
        )

    if missing_labels:
        print("\nImágenes sin archivo de etiqueta:")

        for image_path in missing_labels:
            print(f"  - {image_path.name}")

        raise RuntimeError(
            "Todas las imágenes de esta prueba deben tener "
            "un archivo TXT correspondiente."
        )

    if len(examples) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos imágenes etiquetadas."
        )

    random_generator = random.Random(seed)
    random_generator.shuffle(examples)

    train_count = round(
        len(examples) * train_ratio
    )

    # Garantizamos al menos una imagen en cada conjunto.
    train_count = max(
        1,
        min(train_count, len(examples) - 1),
    )

    train_examples = examples[:train_count]
    validation_examples = examples[train_count:]

    if output_directory.exists():
        print(
            f"Eliminando dataset anterior: "
            f"{output_directory.resolve()}"
        )

        shutil.rmtree(output_directory)

    for image_path, label_path in train_examples:
        copy_example(
            image_path=image_path,
            label_path=label_path,
            output_directory=output_directory,
            split_name="train",
        )

    for image_path, label_path in validation_examples:
        copy_example(
            image_path=image_path,
            label_path=label_path,
            output_directory=output_directory,
            split_name="val",
        )

    yaml_path = create_dataset_yaml(
        output_directory=output_directory,
    )

    print("\nDataset preparado correctamente.")
    print(f"Total de imágenes: {len(examples)}")
    print(f"Entrenamiento: {len(train_examples)}")
    print(f"Validación: {len(validation_examples)}")
    print(f"Carpeta: {output_directory.resolve()}")
    print(f"Configuración: {yaml_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide imágenes y etiquetas YOLO "
            "en entrenamiento y validación."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Carpeta que contiene las imágenes.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Carpeta que contiene las etiquetas TXT.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Carpeta donde se creará el dataset.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporción de entrenamiento. Por defecto: 0.8.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para repetir la misma división.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    prepare_dataset(
        images_directory=args.images,
        labels_directory=args.labels,
        output_directory=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()