"""CMFF + MPC herding strategy inspired by Liu et al. (2026).

This package contains a standalone simulation implementation of the paper's
core algorithmic blocks: collective dynamics, CMFF guidance, and nonlinear MPC.
"""

from .cmff import CMFFGuidance, CMFFResult
from .config import CMFFConfig, DynamicsConfig, MPCConfig, SimulationConfig
from .dynamics import SelfOrganizationRule, step_herd
from .mpc import MPCController, RobotState

__all__ = [
    "CMFFConfig",
    "CMFFGuidance",
    "CMFFResult",
    "DynamicsConfig",
    "MPCConfig",
    "MPCController",
    "RobotState",
    "SelfOrganizationRule",
    "SimulationConfig",
    "step_herd",
]
