from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    """Parameters for the collective motion model in Eq. (2)-(4)."""

    animal_sensing_radius: float = 8.0
    robot_sensing_radius: float = 30.0
    animal_max_speed: float = 2.0
    w_aggregation: float = 0.5
    w_obstacle: float = 0.0
    w_robot: float = 0.5
    w_flee: float = 0.65
    w_threat_cohesion: float = 0.35
    flee_distance_sensitivity: float = 0.015
    boids_separation_distance: float = 2.0
    boids_w_separation: float = 0.20
    boids_w_alignment: float = 0.40
    boids_w_cohesion: float = 0.40
    random_motion_strength: float = 0.35
    epsilon: float = 1e-6


@dataclass(frozen=True)
class CMFFConfig:
    """Parameters for CMFF construction and driving-point generation."""

    sigma: float = 3.0
    divergence_threshold: float = 0.5
    eta: float = 0.5
    driving_offset: float = 8.0
    grid_padding: float = 8.0
    grid_resolution: int = 35
    epsilon: float = 1e-6


@dataclass(frozen=True)
class MPCConfig:
    """Nonlinear MPC parameters for Eq. (16)-(18)."""

    horizon: int = 10
    dt: float = 0.2
    wheelbase: float = 1.4
    min_speed: float = 0.0
    max_speed: float = 8.0
    max_omega: float = 2.0
    max_steering_angle: float = 0.5235987756
    max_acceleration: float = 4.0
    max_angular_acceleration: float = 3.0
    w_driving_point: float = 0.6
    w_centroid_target: float = 0.3
    w_obstacle: float = 0.0
    w_energy: float = 0.1
    obstacle_sensing_radius: float = 25.0
    obstacle_cost_epsilon: float = 0.5
    max_iterations: int = 80


@dataclass(frozen=True)
class SimulationConfig:
    """Default simulation setup from the paper's Section 3.1.1."""

    herd_size: int = 50
    max_steps: int = 600
    initial_region_size: float = 12.0
    target_distance: float = 34.0
    target_radius: float = 5.0
    random_seed: int = 7
