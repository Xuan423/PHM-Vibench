"""Shared modules for the revamped TSPN architecture."""

from .physics_stat_spd import PhysicsStatSPDLayer, PhysicsStatSPDConfig
from .orthogonal_projector import OrthogonalProjector, OrthogonalProjectorConfig
from .heads import ContrastiveHead, ClassifierHead
from .prototype_memory import PrototypeMemory, PrototypeMemoryConfig

__all__ = [
    "PhysicsStatSPDLayer",
    "PhysicsStatSPDConfig",
    "OrthogonalProjector",
    "OrthogonalProjectorConfig",
    "ContrastiveHead",
    "ClassifierHead",
    "PrototypeMemory",
    "PrototypeMemoryConfig",
]
