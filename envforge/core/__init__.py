"""Domain-agnostic core: the environment contract, trajectories, the agent loop, evaluation.

Nothing in this package may import from `envforge.packs`.
"""

from envforge.core.env import (
    Environment,
    Observation,
    Task,
    ToolCall,
    ToolError,
    ToolSpec,
    Verdict,
)
from envforge.core.trajectory import Step, Trajectory

__all__ = [
    "Environment",
    "Observation",
    "Task",
    "ToolCall",
    "ToolError",
    "ToolSpec",
    "Verdict",
    "Step",
    "Trajectory",
]
