from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cmff import CMFFGuidance
from .config import DynamicsConfig, MPCConfig
from .dynamics import SelfOrganizationRule, step_herd

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


@dataclass(frozen=True)
class RobotState:
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega], dtype=float)

    @classmethod
    def from_array(cls, state: np.ndarray) -> "RobotState":
        return cls(float(state[0]), float(state[1]), float(state[2]), float(state[3]), float(state[4]))


class MPCController:
    """Nonlinear MPC solved with SLSQP, matching Eq. (16)-(20)."""

    def __init__(
        self,
        mpc_config: MPCConfig,
        dynamics_config: DynamicsConfig,
        cmff: CMFFGuidance,
        rule: SelfOrganizationRule,
    ):
        self.mpc_config = mpc_config
        self.dynamics_config = dynamics_config
        self.cmff = cmff
        self.rule = rule
        self.previous_solution: np.ndarray | None = None

    def step_robot_state(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        dt = self.mpc_config.dt
        v, omega = control
        return np.array(
            [
                state[0] + v * np.cos(state[2]) * dt,
                state[1] + v * np.sin(state[2]) * dt,
                state[2] + omega * dt,
                v,
                omega,
            ],
            dtype=float,
        )

    def rollout_robot(self, initial_state: RobotState, controls: np.ndarray) -> np.ndarray:
        state = initial_state.as_array()
        states = []
        for control in controls.reshape(-1, 2):
            state = self.step_robot_state(state, control)
            states.append(state.copy())
        return np.asarray(states)

    def visible_obstacles(self, robot_position: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        if obstacles.size == 0:
            return obstacles.reshape(0, 2)
        distances = np.linalg.norm(obstacles - robot_position[None, :], axis=1)
        return obstacles[distances <= self.mpc_config.obstacle_sensing_radius]

    def cost(
        self,
        flat_controls: np.ndarray,
        robot_state: RobotState,
        herd_positions: np.ndarray,
        herd_velocities: np.ndarray,
        target: np.ndarray,
        obstacles: np.ndarray,
        driving_point_override: np.ndarray | None = None,
    ) -> float:
        controls = flat_controls.reshape(-1, 2)
        robot = robot_state.as_array()
        positions = herd_positions.copy()
        velocities = herd_velocities.copy()
        total_driving = 0.0
        total_centroid = 0.0
        total_obstacle = 0.0
        total_energy = 0.0

        for control in controls:
            robot = self.step_robot_state(robot, control)
            positions, velocities = step_herd(
                positions=positions,
                velocities=velocities,
                robot_position=robot[:2],
                obstacles=obstacles,
                rule=self.rule,
                config=self.dynamics_config,
                dt=self.mpc_config.dt,
            )
            cmff_result = self.cmff.compute(positions, velocities, target)
            visible_obstacles = self.visible_obstacles(robot[:2], obstacles)
            driving_point = (
                cmff_result.driving_point
                if driving_point_override is None
                else driving_point_override
            )

            total_driving += float(np.sum((robot[:2] - driving_point) ** 2))
            total_centroid += float(np.sum((cmff_result.centroid - target) ** 2))
            total_energy += float(np.sum(control * control))
            if visible_obstacles.size:
                distances = np.linalg.norm(visible_obstacles - robot[:2], axis=1)
                total_obstacle += float(np.sum(1.0 / (distances ** 2 + self.mpc_config.obstacle_cost_epsilon)))

        return (
            self.mpc_config.w_driving_point * total_driving
            + self.mpc_config.w_centroid_target * total_centroid
            + self.mpc_config.w_obstacle * total_obstacle
            + self.mpc_config.w_energy * total_energy
        )

    def bounds(self) -> list[tuple[float, float]]:
        omega_limit = min(
            self.mpc_config.max_omega,
            self.mpc_config.max_speed / self.mpc_config.wheelbase * np.tan(self.mpc_config.max_steering_angle),
        )
        return [
            (self.mpc_config.min_speed, self.mpc_config.max_speed),
            (-omega_limit, omega_limit),
        ] * self.mpc_config.horizon

    def rate_constraints(self, previous_control: np.ndarray) -> list[dict]:
        dt = self.mpc_config.dt
        max_dv = self.mpc_config.max_acceleration * dt
        max_dw = self.mpc_config.max_angular_acceleration * dt
        constraints = []

        def control_at(flat: np.ndarray, index: int) -> np.ndarray:
            return flat.reshape(-1, 2)[index]

        for k in range(self.mpc_config.horizon):
            for component, max_delta in ((0, max_dv), (1, max_dw)):
                reference_index = k - 1
                if k == 0:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda flat, c=component, md=max_delta: md - abs(control_at(flat, 0)[c] - previous_control[c]),
                        }
                    )
                else:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda flat, c=component, md=max_delta, i=k, j=reference_index: md
                            - abs(control_at(flat, i)[c] - control_at(flat, j)[c]),
                        }
                    )
        return constraints

    def initial_guess(self, robot_state: RobotState) -> np.ndarray:
        if self.previous_solution is not None:
            shifted = np.vstack((self.previous_solution.reshape(-1, 2)[1:], self.previous_solution.reshape(-1, 2)[-1:]))
            return shifted.ravel()
        guess = np.tile(np.array([max(robot_state.v, 0.5), robot_state.omega], dtype=float), self.mpc_config.horizon)
        lower_upper = self.bounds()
        return np.array([np.clip(value, low, high) for value, (low, high) in zip(guess, lower_upper)])

    def fallback_control(self, robot_state: RobotState, driving_point: np.ndarray) -> np.ndarray:
        delta = driving_point - robot_state.position
        desired_theta = np.arctan2(delta[1], delta[0])
        angle_error = np.arctan2(np.sin(desired_theta - robot_state.theta), np.cos(desired_theta - robot_state.theta))
        speed = min(self.mpc_config.max_speed, max(0.3, np.linalg.norm(delta)))
        omega = np.clip(2.0 * angle_error, -self.mpc_config.max_omega, self.mpc_config.max_omega)
        return np.array([speed, omega], dtype=float)

    def solve(
        self,
        robot_state: RobotState,
        herd_positions: np.ndarray,
        herd_velocities: np.ndarray,
        target: np.ndarray,
        obstacles: np.ndarray,
        driving_point_override: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool, float]:
        cmff_result = self.cmff.compute(herd_positions, herd_velocities, target)
        if minimize is None:
            return self.fallback_control(robot_state, cmff_result.driving_point), False, float("nan")

        previous_control = np.array([robot_state.v, robot_state.omega], dtype=float)
        result = minimize(
            fun=self.cost,
            x0=self.initial_guess(robot_state),
            args=(robot_state, herd_positions, herd_velocities, target, obstacles, driving_point_override),
            method="SLSQP",
            bounds=self.bounds(),
            constraints=self.rate_constraints(previous_control),
            options={"maxiter": self.mpc_config.max_iterations, "ftol": 1e-4, "disp": False},
        )

        if result.success:
            self.previous_solution = result.x.copy()
            return result.x.reshape(-1, 2)[0], True, float(result.fun)

        fallback_point = cmff_result.driving_point if driving_point_override is None else driving_point_override
        control = self.fallback_control(robot_state, fallback_point)
        return control, False, float(result.fun) if np.isfinite(result.fun) else float("nan")
