from pathlib import Path

from ultralytics import settings


def main() -> None:
    # config_yolo.py está dentro de src, por eso se utiliza parent.parent.
    project_dir = Path(__file__).resolve().parent.parent

    datasets_dir = project_dir / "datasets"
    models_dir = project_dir / "models"
    runs_dir = project_dir / "runs"

    # Crea las carpetas si todavía no existen.
    for directory in (datasets_dir, models_dir, runs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Guarda permanentemente estas rutas en la configuración de Ultralytics.
    settings.update(
        {
            "datasets_dir": str(datasets_dir),
            "weights_dir": str(models_dir),
            "runs_dir": str(runs_dir),
        }
    )

    print("Configuración de Ultralytics guardada correctamente.")
    print(f"Datasets:   {settings['datasets_dir']}")
    print(f"Modelos:    {settings['weights_dir']}")
    print(f"Resultados: {settings['runs_dir']}")


if __name__ == "__main__":
    main()