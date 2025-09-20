"""Contrastive wrapper around the baseline TSPN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .TSPN import Model as TSPNModel


@dataclass
class ContrastiveConfig:
    enabled: bool
    projection_hidden: int
    projection_dim: int
    temperature: float
    loss_weight: float
    mode: str = "supervised"
    return_embeddings: bool = True


def _to_contrastive_config(raw_cfg: Any) -> ContrastiveConfig:
    """Convert configuration namespace into a structured dataclass."""
    if raw_cfg is None:
        return ContrastiveConfig(False, 0, 0, 1.0, 0.0)

    enabled = bool(getattr(raw_cfg, "enabled", False))
    projection_hidden = int(getattr(raw_cfg, "projection_hidden", 0))
    projection_dim = int(getattr(raw_cfg, "projection_dim", 0))
    temperature = float(getattr(raw_cfg, "temperature", 1.0))
    loss_weight = float(getattr(raw_cfg, "loss_weight", 0.0))
    mode = getattr(raw_cfg, "mode", "supervised")
    return_embeddings = bool(getattr(raw_cfg, "return_embeddings", True))

    return ContrastiveConfig(
        enabled=enabled,
        projection_hidden=projection_hidden,
        projection_dim=projection_dim,
        temperature=temperature,
        loss_weight=loss_weight,
        mode=mode,
        return_embeddings=return_embeddings,
    )


class Model(nn.Module):
    """TSPN backbone with an optional contrastive projection head.

    The wrapper leaves the baseline behaviour untouched when the
    contrastive configuration is disabled. When enabled, a projector is
    provided and the forward method can optionally return embeddings for
    contrastive loss computation.
    """

    def __init__(self, args: Any, metadata: Optional[Any] = None) -> None:
        super().__init__()
        self.args = args
        self.backbone = TSPNModel(args, metadata)

        raw_cfg = getattr(args, "contrastive", None)
        self.contrastive_cfg = _to_contrastive_config(raw_cfg)

        self.feature_dim = getattr(self.backbone, "channel_for_classifier", None)
        if self.feature_dim is None:
            raise ValueError("TSPNContrastive requires backbone to expose channel_for_classifier")

        self.temperature = self.contrastive_cfg.temperature
        self.contrastive_weight = self.contrastive_cfg.loss_weight
        self.mode = self.contrastive_cfg.mode

        self.projection_head: Optional[nn.Module] = None
        if self.contrastive_cfg.enabled:
            self._validate_contrastive_config()
            self.projection_head = self._build_projection_head()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _validate_contrastive_config(self) -> None:
        cfg = self.contrastive_cfg
        missing: Dict[str, Any] = {}
        if cfg.projection_dim <= 0:
            missing["projection_dim"] = cfg.projection_dim
        if cfg.projection_hidden < 0:
            missing["projection_hidden"] = cfg.projection_hidden
        if cfg.temperature <= 0:
            missing["temperature"] = cfg.temperature
        if not missing:
            return
        raise ValueError(
            "Invalid contrastive configuration for TSPNContrastive: "
            + ", ".join(f"{key}={value}" for key, value in missing.items())
        )

    def _build_projection_head(self) -> nn.Module:
        cfg = self.contrastive_cfg
        layers = []
        hidden = cfg.projection_hidden
        proj_dim = cfg.projection_dim

        if hidden and hidden > 0:
            layers.append(nn.Linear(self.feature_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden, proj_dim))
        else:
            layers.append(nn.Linear(self.feature_dim, proj_dim))

        projector = nn.Sequential(*layers)
        device = getattr(self.args, "device", "cpu")
        return projector.to(device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the flattened feature representation prior to classification."""
        out = x
        for layer in self.backbone.signal_processing_layers:
            out = layer(out)
        features = self.backbone.feature_extractor_layers(out)
        return features

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the contrastive projection head if it exists."""
        if self.projection_head is None:
            return embeddings
        return self.projection_head(embeddings)

    # ------------------------------------------------------------------
    # Forward interfaces
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        data_id: Optional[Any] = None,
        task_id: Optional[Any] = None,
        *,
        return_embeddings: bool = False,
    ) -> Any:
        if not return_embeddings:
            return self.backbone.forward(x, data_id=data_id, task_id=task_id)

        embeddings = self.encode(x)
        logits = self.backbone.clf(embeddings)
        projected = self.project(embeddings)
        return {
            "logits": logits,
            "embeddings": embeddings,
            "projection": projected,
            "temperature": self.temperature,
        }

    # The default classification loss path expects tensors, so expose a dedicated
    # helper that mirrors the original backbone interface when only logits are
    # required.
    def forward_logits(
        self,
        x: torch.Tensor,
        data_id: Optional[Any] = None,
        task_id: Optional[Any] = None,
    ) -> torch.Tensor:
        return self.backbone.forward(x, data_id=data_id, task_id=task_id)
