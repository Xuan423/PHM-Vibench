"""Callback utilities for trainer factory implementations."""

from .contrastive_logging import ContrastiveTelemetryCallback
from .projector_trajectory import ProjectorTrajectoryLogger
from .prototype_checkpoint import PrototypeCheckpointCallback

__all__ = [
    "ContrastiveTelemetryCallback",
    "ProjectorTrajectoryLogger",
    "PrototypeCheckpointCallback",
]
