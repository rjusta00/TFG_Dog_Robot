from __future__ import annotations

import math


METRIC_COLUMNS = [
    "metric_method",
    "metric_valid",
    "dog_track_id",
    "dog_x_px",
    "dog_y_px",
    "flock_x_px",
    "flock_y_px",
    "mpc_point_x_px",
    "mpc_point_y_px",
    "flock_scale_px",
    "dog_flock_distance_px",
    "mpc_dog_distance_px",
    "mpc_flock_distance_px",
    "dog_flock_distance_flock_units",
    "mpc_dog_distance_flock_units",
    "mpc_flock_distance_flock_units",
]


def select_closest_dog(dogs: list[dict], point: tuple[float, float] | None) -> dict | None:
    if point is None or not dogs:
        return None

    return min(
        dogs,
        key=lambda dog: math.hypot(
            float(dog["center"][0]) - point[0],
            float(dog["center"][1]) - point[1],
        ),
    )


def build_spatial_metrics(
    method: str,
    dog: dict | None,
    flock_center: tuple[float, float] | None,
    mpc_point: tuple[float, float] | None,
    flock_scale: float | None,
) -> dict[str, str | int]:
    """Return CSV-safe metrics; distances are blank when a current dog is unavailable."""
    row: dict[str, str | int] = {column: "" for column in METRIC_COLUMNS}
    row["metric_method"] = method
    row["metric_valid"] = 0

    if flock_center is not None:
        row["flock_x_px"] = f"{flock_center[0]:.3f}"
        row["flock_y_px"] = f"{flock_center[1]:.3f}"
    if mpc_point is not None:
        row["mpc_point_x_px"] = f"{mpc_point[0]:.3f}"
        row["mpc_point_y_px"] = f"{mpc_point[1]:.3f}"
    if flock_scale is not None and flock_scale > 0.0:
        row["flock_scale_px"] = f"{flock_scale:.3f}"

    if dog is None or flock_center is None or mpc_point is None or flock_scale is None or flock_scale <= 0.0:
        return row

    dog_x, dog_y = float(dog["center"][0]), float(dog["center"][1])
    dog_flock = math.hypot(dog_x - flock_center[0], dog_y - flock_center[1])
    mpc_dog = math.hypot(mpc_point[0] - dog_x, mpc_point[1] - dog_y)
    mpc_flock = math.hypot(mpc_point[0] - flock_center[0], mpc_point[1] - flock_center[1])

    row.update(
        {
            "metric_valid": 1,
            "dog_track_id": "" if dog.get("track_id") is None else int(dog["track_id"]),
            "dog_x_px": f"{dog_x:.3f}",
            "dog_y_px": f"{dog_y:.3f}",
            "dog_flock_distance_px": f"{dog_flock:.3f}",
            "mpc_dog_distance_px": f"{mpc_dog:.3f}",
            "mpc_flock_distance_px": f"{mpc_flock:.3f}",
            "dog_flock_distance_flock_units": f"{dog_flock / flock_scale:.6f}",
            "mpc_dog_distance_flock_units": f"{mpc_dog / flock_scale:.6f}",
            "mpc_flock_distance_flock_units": f"{mpc_flock / flock_scale:.6f}",
        }
    )
    return row
