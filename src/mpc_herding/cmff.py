from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CMFFConfig


@dataclass(frozen=True)
class CMFFResult:
    centroid: np.ndarray
    q_div: np.ndarray
    driving_point: np.ndarray
    d_escape: np.ndarray
    d_herd: np.ndarray
    fused_direction: np.ndarray
    max_divergence: float
    escape_detected: bool


class CMFFGuidance:
    """Collective motion flow field guidance from Eq. (6)-(13)."""

    def __init__(self, config: CMFFConfig):
        self.config = config

    def gaussian_kernel(self, query_points: np.ndarray, positions: np.ndarray) -> np.ndarray:
        rho = query_points[:, None, :] - positions[None, :, :]
        sq_norm = np.sum(rho * rho, axis=2)
        sigma2 = self.config.sigma ** 2
        return np.exp(-sq_norm / (2.0 * sigma2)) / (2.0 * np.pi * sigma2)

    def flow_at(self, query_points: np.ndarray, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        kernel = self.gaussian_kernel(query_points, positions)
        return kernel @ velocities / max(len(positions), 1)

    def divergence_at(self, query_points: np.ndarray, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        rho = query_points[:, None, :] - positions[None, :, :]
        kernel = self.gaussian_kernel(query_points, positions)
        dot = np.sum(velocities[None, :, :] * rho, axis=2)
        return -np.sum(kernel * dot, axis=1) / (max(len(positions), 1) * self.config.sigma ** 2)

    def make_query_grid(self, positions: np.ndarray) -> np.ndarray:
        lower = positions.min(axis=0) - self.config.grid_padding
        upper = positions.max(axis=0) + self.config.grid_padding
        xs = np.linspace(lower[0], upper[0], self.config.grid_resolution)
        ys = np.linspace(lower[1], upper[1], self.config.grid_resolution)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def compute(self, positions: np.ndarray, velocities: np.ndarray, target: np.ndarray) -> CMFFResult:
        centroid = positions.mean(axis=0)
        query_points = self.make_query_grid(positions)
        divergence = self.divergence_at(query_points, positions, velocities)

        escape_mask = divergence > self.config.divergence_threshold
        if np.any(escape_mask):
            escape_indices = np.flatnonzero(escape_mask)
            best_index = escape_indices[np.argmax(divergence[escape_indices])]
            escape_detected = True
        else:
            best_index = int(np.argmax(divergence))
            escape_detected = False

        q_div = query_points[best_index]
        flow = self.flow_at(q_div[None, :], positions, velocities)[0]
        d_escape = flow / max(np.linalg.norm(flow), self.config.epsilon)

        d_herd_raw = -(target - centroid)
        d_herd = d_herd_raw / max(np.linalg.norm(d_herd_raw), self.config.epsilon)

        fused = self.config.eta * d_escape + (1.0 - self.config.eta) * d_herd
        fused_direction = fused / max(np.linalg.norm(fused), self.config.epsilon)
        driving_point = q_div - self.config.driving_offset * fused_direction

        return CMFFResult(
            centroid=centroid,
            q_div=q_div,
            driving_point=driving_point,
            d_escape=d_escape,
            d_herd=d_herd,
            fused_direction=fused_direction,
            max_divergence=float(divergence[best_index]),
            escape_detected=escape_detected,
        )
