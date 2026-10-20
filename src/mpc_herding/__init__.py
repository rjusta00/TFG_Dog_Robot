"""MPC herding strategy inspired by Liu et al. (2026).

This package contains a standalone simulation implementation of the paper's
core algorithmic blocks: collective dynamics and nonlinear MPC.
"""

from .config import DynamicsConfig, MPCConfig, SimulationConfig
from .dynamics import SelfOrganizationRule, step_herd
from .mpc import MPCController, RobotState

__all__ = [
    "DynamicsConfig",
    "MPCConfig",
    "MPCController",
    "RobotState",
    "SelfOrganizationRule",
    "SimulationConfig",
    "step_herd",
]
