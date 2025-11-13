"""Shared modules for physics-conditioned TSPN variants."""

from .physics_convex_projector import PhysicsConvexProjector, SimplexStats
from .spd_metric import SPDCouplingMetric, SPDCouplingState

__all__ = [
    "PhysicsConvexProjector",
    "SimplexStats",
    "SPDCouplingMetric",
    "SPDCouplingState",
]
