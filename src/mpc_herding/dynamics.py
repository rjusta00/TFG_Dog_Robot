from __future__ import annotations

from enum import Enum

import numpy as np

from .config import DynamicsConfig


class SelfOrganizationRule(str, Enum):
    """Self-organization rules discussed in the paper."""

    RANDOM = "ran"
    NEAREST_NEIGHBOR = "nn"
    HAMILTONIAN = "ha"
    LOCAL_CROWDED_HORIZON = "lch"
    NON_LOCAL_AGGREGATION = "nla"
    VICSEK = "vicsek"
    BOIDS = "boids"


def normalize_vectors(vectors: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, epsilon)


def limit_vector_norms(vectors: np.ndarray, max_norm: float, epsilon: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, epsilon))
    return vectors * scale


def pairwise_offsets(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(offsets, axis=2)
    return offsets, distances


def neighbor_masks(positions: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets, distances = pairwise_offsets(positions)
    mask = (distances <= radius) & (distances > 0.0)
    return offsets, distances, mask


def aggregation_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    rule: SelfOrganizationRule,
    config: DynamicsConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    offsets, distances, mask = neighbor_masks(positions, config.animal_sensing_radius)
    n = len(positions)
    result = np.zeros((n, 2), dtype=float)
    rng = rng or np.random.default_rng()

    if rule == SelfOrganizationRule.RANDOM:
        angles = rng.uniform(-np.pi, np.pi, size=n)
        return config.random_motion_strength * np.column_stack((np.cos(angles), np.sin(angles)))

    for i in range(n):
        neighbor_indices = np.flatnonzero(mask[i])
        if len(neighbor_indices) == 0:
            continue

        neighbor_offsets = offsets[i, neighbor_indices]
        neighbor_distances = distances[i, neighbor_indices]

        if rule == SelfOrganizationRule.NEAREST_NEIGHBOR:
            nearest = neighbor_indices[np.argmin(neighbor_distances)]
            result[i] = positions[nearest] - positions[i]

        elif rule == SelfOrganizationRule.HAMILTONIAN:
            if len(neighbor_indices) == 1:
                result[i] = positions[neighbor_indices[0]] - positions[i]
            else:
                two = neighbor_indices[np.argsort(neighbor_distances)[:2]]
                gap_midpoint = 0.5 * (positions[two[0]] + positions[two[1]])
                result[i] = gap_midpoint - positions[i]

        elif rule == SelfOrganizationRule.LOCAL_CROWDED_HORIZON:
            weights = 1.0 / np.maximum(neighbor_distances, config.epsilon)
            local_center = np.average(positions[neighbor_indices], axis=0, weights=weights)
            result[i] = local_center - positions[i]

        elif rule == SelfOrganizationRule.NON_LOCAL_AGGREGATION:
            weights = neighbor_distances / np.maximum(neighbor_distances.sum(), config.epsilon)
            local_center = np.average(positions[neighbor_indices], axis=0, weights=weights)
            result[i] = local_center - positions[i]

        elif rule == SelfOrganizationRule.VICSEK:
            mean_velocity = velocities[neighbor_indices].mean(axis=0)
            if np.linalg.norm(mean_velocity) < config.epsilon:
                mean_velocity = positions[neighbor_indices].mean(axis=0) - positions[i]
            result[i] = mean_velocity

        elif rule == SelfOrganizationRule.BOIDS:
            close = neighbor_distances < config.boids_separation_distance
            separation = np.zeros(2, dtype=float)
            if np.any(close):
                separation = -normalize_vectors(neighbor_offsets[close], config.epsilon).sum(axis=0)

            alignment = velocities[neighbor_indices].mean(axis=0)
            cohesion = positions[neighbor_indices].mean(axis=0) - positions[i]
            result[i] = (
                config.boids_w_separation * separation
                + config.boids_w_alignment * alignment
                + config.boids_w_cohesion * cohesion
            )

    return normalize_vectors(result, config.epsilon)


def obstacle_avoidance_velocity(
    positions: np.ndarray,
    obstacles: np.ndarray,
    config: DynamicsConfig,
) -> np.ndarray:
    if obstacles.size == 0:
        return np.zeros_like(positions)

    result = np.zeros_like(positions, dtype=float)
    for i, position in enumerate(positions):
        obstacle_offsets = position[None, :] - obstacles
        distances = np.linalg.norm(obstacle_offsets, axis=1)
        local = distances <= config.animal_sensing_radius
        if np.any(local):
            result[i] = (obstacle_offsets[local] / np.maximum(distances[local, None], config.epsilon)).sum(axis=0)
    return normalize_vectors(result, config.epsilon)


def robot_response_velocity(
    positions: np.ndarray,
    robot_position: np.ndarray,
    config: DynamicsConfig,
) -> np.ndarray:
    result = np.zeros_like(positions, dtype=float)
    offsets_from_robot = positions - robot_position[None, :]
    distances = np.linalg.norm(offsets_from_robot, axis=1)
    threatened = distances <= config.robot_sensing_radius
    _, _, neighbor_mask = neighbor_masks(positions, config.animal_sensing_radius)

    for i in np.flatnonzero(threatened):
        flee = offsets_from_robot[i] / max(distances[i], config.epsilon)
        flee *= np.exp(-config.flee_distance_sensitivity * distances[i] ** 2)

        neighbors = np.flatnonzero(neighbor_mask[i])
        if len(neighbors) == 0:
            cohesion = np.zeros(2, dtype=float)
        else:
            local_centroid = positions[neighbors].mean(axis=0)
            cohesion = local_centroid - positions[i]
            cohesion = cohesion / max(np.linalg.norm(cohesion), config.epsilon)

        result[i] = config.w_flee * flee + config.w_threat_cohesion * cohesion

    return normalize_vectors(result, config.epsilon)


def collective_velocity(
    positions: np.ndarray,
    velocities: np.ndarray,
    robot_position: np.ndarray,
    obstacles: np.ndarray,
    rule: SelfOrganizationRule,
    config: DynamicsConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    f_agg = aggregation_velocity(positions, velocities, rule, config, rng)
    f_obs = obstacle_avoidance_velocity(positions, obstacles, config)
    f_robot = robot_response_velocity(positions, robot_position, config)
    combined = (
        config.w_aggregation * f_agg
        + config.w_obstacle * f_obs
        + config.w_robot * f_robot
    )
    return limit_vector_norms(combined, config.animal_max_speed, config.epsilon)


def step_herd(
    positions: np.ndarray,
    velocities: np.ndarray,
    robot_position: np.ndarray,
    obstacles: np.ndarray,
    rule: SelfOrganizationRule,
    config: DynamicsConfig,
    dt: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    new_velocities = collective_velocity(
        positions=positions,
        velocities=velocities,
        robot_position=robot_position,
        obstacles=obstacles,
        rule=rule,
        config=config,
        rng=rng,
    )
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities
