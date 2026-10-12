from __future__ import annotations

import subprocess
import threading
import time


GPU_TELEMETRY_COLUMNS = (
    "gpu_power_w",
    "gpu_memory_used_mb",
    "gpu_utilization_pct",
    "gpu_temperature_c",
    "gpu_energy_j",
)


def gpu_index(device: str | int) -> int:
    """Return the NVIDIA device index from values such as ``0`` or ``cuda:0``."""
    text = str(device)
    return int(text.rsplit(":", maxsplit=1)[-1]) if ":" in text else int(text)


class GpuTelemetryMonitor:
    """Sample NVIDIA GPU telemetry outside the control loop."""

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self._latest = {column: 0.0 for column in GPU_TELEMETRY_COLUMNS}
        self._energy_j = 0.0
        self._last_power_w: float | None = None
        self._last_sample_time: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = False

    def _poll(self) -> bool:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.device_index),
                    "--query-gpu=power.draw,memory.used,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                text=True,
                timeout=2,
            )
            values = [float(value.strip()) for value in result.stdout.strip().split(",")]
            if len(values) != 4:
                return False
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            return False

        now = time.perf_counter()
        with self._lock:
            if self._last_sample_time is not None and self._last_power_w is not None:
                self._energy_j += (self._last_power_w + values[0]) * 0.5 * (now - self._last_sample_time)
            self._last_power_w = values[0]
            self._last_sample_time = now
            self._latest = {
                "gpu_power_w": values[0],
                "gpu_memory_used_mb": values[1],
                "gpu_utilization_pct": values[2],
                "gpu_temperature_c": values[3],
                "gpu_energy_j": self._energy_j,
            }
        return True

    def start(self) -> None:
        self.available = self._poll()
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._poll()

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return self._latest.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)
        if self.available:
            self._poll()
