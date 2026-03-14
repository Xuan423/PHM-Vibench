"""Interpretable time-frequency multi-prototype DG model package."""

from .config_schema import TFMultiProtoDGConfig, build_model_config
from .model import Model

__all__ = ["Model", "TFMultiProtoDGConfig", "build_model_config"]
