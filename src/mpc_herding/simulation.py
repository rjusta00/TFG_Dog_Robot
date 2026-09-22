from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .cmff import CMFFGuidance
from .config import CMFFConfig, DynamicsConfig, MPCConfig, SimulationConfig
from .dynamics import SelfOrganizationRule, step_herd
from .mpc import MPCController, RobotState


def initialize_herd(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, RobotState]:
    rng = np.random.default_rng(config.random_seed)
    half = config.initial_region_size / 2.0
    positions = rng.uniform(-half, half, size=(config.herd_size, 2))
    velocities = np.zeros_like(positions)
    target = np.array([config.target_distance / np.sqrt(2.0), config.target_distance / np.sqrt(2.0)])
    robot_state = RobotState(x=-12.0, y=-12.0, theta=np.pi / 4.0)
    return positions, velocities, target, robot_state


def coverage_rate(positions: np.ndarray, target: np.ndarray, radius: float) -> float:
    distances = np.linalg.norm(positions - target[None, :], axis=1)
    return float(np.mean(distances <= radius))


def collective_dispersion(positions: np.ndarray) -> float:
    centroid = positions.mean(axis=0)
    return float(np.mean(np.linalg.norm(positions - centroid[None, :], axis=1)))


def run_simulation(
    simulation_config: SimulationConfig,
    dynamics_config: DynamicsConfig,
    cmff_config: CMFFConfig,
    mpc_config: MPCConfig,
    rule: SelfOrganizationRule,
    obstacles: np.ndarray,
    output_csv: Path | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(simulation_config.random_seed + 1)
    positions, velocities, target, robot_state = initialize_herd(simulation_config)
    cmff = CMFFGuidance(cmff_config)
    controller = MPCController(mpc_config, dynamics_config, cmff, rule)
    robot_distance = 0.0
    herd_distance = np.zeros(simulation_config.herd_size, dtype=float)
    rows = []

    for step in range(simulation_config.max_steps):
        guidance = cmff.compute(positions, velocities, target)
        control, optimizer_success, optimizer_cost = controller.solve(robot_state, positions, velocities, target, obstacles)
        previous_robot_position = robot_state.position
        new_robot_array = controller.step_robot_state(robot_state.as_array(), control)
        robot_state = RobotState.from_array(new_robot_array)
        robot_distance += float(np.linalg.norm(robot_state.position - previous_robot_position))

        previous_positions = positions.copy()
        positions, velocities = step_herd(
            positions=positions,
            velocities=velocities,
            robot_position=robot_state.position,
            obstacles=obstacles,
            rule=rule,
            config=dynamics_config,
            dt=mpc_config.dt,
            rng=rng,
        )
        herd_distance += np.linalg.norm(positions - previous_positions, axis=1)

        current_coverage = coverage_rate(positions, target, simulation_config.target_radius)
        rows.append(
            {
                "step": step,
                "robot_x": robot_state.x,
                "robot_y": robot_state.y,
                "robot_theta": robot_state.theta,
                "control_v": control[0],
                "control_omega": control[1],
                "centroid_x": guidance.centroid[0],
                "centroid_y": guidance.centroid[1],
                "target_x": target[0],
                "target_y": target[1],
                "driving_point_x": guidance.driving_point[0],
                "driving_point_y": guidance.driving_point[1],
                "q_div_x": guidance.q_div[0],
                "q_div_y": guidance.q_div[1],
                "max_divergence": guidance.max_divergence,
                "escape_detected": int(guidance.escape_detected),
                "coverage_rate": current_coverage,
                "collective_dispersion": collective_dispersion(positions),
                "optimizer_success": int(optimizer_success),
                "optimizer_cost": optimizer_cost,
            }
        )

        if current_coverage >= 1.0:
            break

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {
        "success": float(coverage_rate(positions, target, simulation_config.target_radius) >= 1.0),
        "coverage_rate": coverage_rate(positions, target, simulation_config.target_radius),
        "completion_steps": float(len(rows)),
        "robot_travel_distance": robot_distance,
        "herd_average_travel_distance": float(np.mean(herd_distance)),
        "collective_dispersion": collective_dispersion(positions),
    }


def parse_obstacles(values: list[str]) -> np.ndarray:
    obstacles = []
    for value in values:
        x_text, y_text = value.split(",", maxsplit=1)
        obstacles.append((float(x_text), float(y_text)))
    return np.asarray(obstacles, dtype=float).reshape(-1, 2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CMFF + MPC herding simulation.")
    parser.add_argument("--herd-size", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--rule", choices=[rule.value for rule in SelfOrganizationRule], default=SelfOrganizationRule.BOIDS.value)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--target-radius", type=float, default=5.0)
    parser.add_argument("--obstacle", action="append", default=[], help="Obstacle as x,y. Can be repeated.")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation_config = SimulationConfig(
        herd_size=args.herd_size,
        max_steps=args.max_steps,
        target_radius=args.target_radius,
        random_seed=args.seed,
    )
    dynamics_config = DynamicsConfig(w_obstacle=0.1 if args.obstacle else 0.0)
    cmff_config = CMFFConfig(eta=args.eta, sigma=args.sigma, divergence_threshold=args.tau)
    mpc_config = MPCConfig(horizon=args.horizon, w_obstacle=0.05 if args.obstacle else 0.0)
    metrics = run_simulation(
        simulation_config=simulation_config,
        dynamics_config=dynamics_config,
        cmff_config=cmff_config,
        mpc_config=mpc_config,
        rule=SelfOrganizationRule(args.rule),
        obstacles=parse_obstacles(args.obstacle),
        output_csv=args.output_csv,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
