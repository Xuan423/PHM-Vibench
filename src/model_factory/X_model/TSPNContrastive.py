"""Contrastive wrapper around the baseline TSPN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .TSPN import Model as TSPNModel
from ..modules import PhysicsConvexProjector, SPDCouplingMetric


@dataclass
class ProjectorPriorConfig:
    path: Optional[str] = None
    format: str = "npy"
    temperature: float = 1.0


@dataclass
class PhysicalProjectorConfig:
    enabled: bool = False
    indicator_dim: int = 64
    init: str = "uniform"
    simplex_eps: float = 1e-6
    spd_rank: int = 4
    tau_init: float = 1.0
    tau_min: float = 1e-4
    prior: ProjectorPriorConfig = field(default_factory=ProjectorPriorConfig)


@dataclass
class ContrastiveConfig:
    enabled: bool
    temperature: float
    loss_weight: float
    reg_weight: float = 0.0
    physical_projector: PhysicalProjectorConfig = field(default_factory=PhysicalProjectorConfig)


def _to_physical_projector_config(raw_cfg: Any) -> PhysicalProjectorConfig:
    if raw_cfg is None:
        return PhysicalProjectorConfig()

    prior_raw = getattr(raw_cfg, "prior", None)
    prior_cfg = ProjectorPriorConfig(
        path=getattr(prior_raw, "path", None) if prior_raw is not None else None,
        format=str(getattr(prior_raw, "format", "npy") or "npy") if prior_raw is not None else "npy",
        temperature=float(getattr(prior_raw, "temperature", 1.0)) if prior_raw is not None else 1.0,
    )

    return PhysicalProjectorConfig(
        enabled=bool(getattr(raw_cfg, "enabled", False)),
        indicator_dim=int(getattr(raw_cfg, "indicator_dim", 64)),
        init=str(getattr(raw_cfg, "init", "uniform")),
        simplex_eps=float(getattr(raw_cfg, "simplex_eps", 1e-6)),
        spd_rank=int(getattr(raw_cfg, "spd_rank", 4)),
        tau_init=float(getattr(raw_cfg, "tau_init", 1.0)),
        tau_min=float(getattr(raw_cfg, "tau_min", 1e-4)),
        prior=prior_cfg,
    )


def _to_contrastive_config(raw_cfg: Any) -> ContrastiveConfig:
    """Convert configuration namespace into a structured dataclass."""
    if raw_cfg is None:
        return ContrastiveConfig(False, 1.0, 0.0, 0.0)

    enabled = bool(getattr(raw_cfg, "enabled", False))
    temperature = float(getattr(raw_cfg, "temperature", 1.0))
    loss_weight = float(getattr(raw_cfg, "loss_weight", 0.0))
    reg_weight = float(getattr(raw_cfg, "reg_weight", getattr(raw_cfg, "lambda_reg", 0.0)))
    physical_projector = _to_physical_projector_config(getattr(raw_cfg, "physical_projector", None))

    return ContrastiveConfig(
        enabled=enabled,
        temperature=temperature,
        loss_weight=loss_weight,
        reg_weight=reg_weight,
        physical_projector=physical_projector,
    )


class IndicatorClassifier(nn.Module):
    """Classifier operating on physical indicator vectors with explainable weights."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, indicators: torch.Tensor) -> torch.Tensor:
        flat = indicators.view(indicators.size(0), -1)
        return self.linear(flat)

    def explain_weights(self) -> Dict[str, torch.Tensor]:
        weight = self.linear.weight.detach().clone()
        bias = self.linear.bias.detach().clone() if self.linear.bias is not None else None
        payload: Dict[str, torch.Tensor] = {"weight": weight}
        if bias is not None:
            payload["bias"] = bias
        return payload


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
        self._last_projector_penalty: Dict[str, torch.Tensor] = {}

        device = getattr(self.args, "device", "cpu")
        self.projector_cfg = self.contrastive_cfg.physical_projector
        self.physics_projector: Optional[PhysicsConvexProjector] = None
        self.spd_metric: Optional[SPDCouplingMetric] = None
        self.classifier: nn.Module = self.backbone.clf
        self._embedding_dim = self.feature_dim

        if self.projector_cfg.enabled:
            self._initialize_physics_layers(device=device)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _initialize_physics_layers(self, *, device: Optional[Any]) -> None:
        if self.projector_cfg.indicator_dim <= 0:
            raise ValueError("Physics projector requires indicator_dim > 0")
        num_classes = getattr(self.backbone, "num_classes", None)
        if num_classes is None:
            num_classes = int(getattr(self.args, "num_classes", 0))
        if num_classes <= 0:
            raise ValueError("Physics projector requires a valid num_classes value.")

        projector = self._build_physics_projector(device=device)
        metric = SPDCouplingMetric(
            axis_dim=self.projector_cfg.indicator_dim,
            rank=max(1, self.projector_cfg.spd_rank),
            tau_init=self.projector_cfg.tau_init,
            tau_min=self.projector_cfg.tau_min,
        )

        self.physics_projector = projector.to(device)
        self.spd_metric = metric.to(device)
        self.classifier = IndicatorClassifier(self.projector_cfg.indicator_dim, num_classes).to(device)
        self._embedding_dim = self.projector_cfg.indicator_dim

    def _build_physics_projector(self, *, device: Optional[Any]) -> PhysicsConvexProjector:
        mask = self._load_projector_prior_mask(device=device)
        projector = PhysicsConvexProjector(
            input_dim=self.feature_dim,
            axis_dim=self.projector_cfg.indicator_dim,
            simplex_eps=self.projector_cfg.simplex_eps,
            prior_mask=mask,
            init=self.projector_cfg.init,
        )
        return projector

    def _load_projector_prior_mask(self, *, device: Optional[Any]) -> Optional[torch.Tensor]:
        prior_cfg = self.projector_cfg.prior
        if prior_cfg.path is None:
            return None

        path = Path(prior_cfg.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Physical projector prior not found: {path}")

        fmt = (prior_cfg.format or "npy").lower()
        if fmt == "npy":
            data = np.load(path)
        elif fmt in ("yaml", "yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - guard for missing optional dependency
                raise ImportError("PyYAML is required to load YAML projector priors.") from exc
            with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported projector prior format: {fmt}")

        mask = torch.as_tensor(data, dtype=torch.float32)

        expected = (self.feature_dim, self.projector_cfg.indicator_dim)
        if mask.shape != expected:
            raise ValueError(
                f"Physical projector prior shape {mask.shape} does not match expected {expected}."
            )

        if prior_cfg.temperature != 1.0:
            mask = torch.sigmoid(prior_cfg.temperature * (mask - 0.5))

        return mask.to(device)

    def encode(self, x: torch.Tensor, *, return_raw: bool = False) -> torch.Tensor:
        """Return indicator embeddings (or raw features when requested)."""
        out = x
        for layer in self.backbone.signal_processing_layers:
            out = layer(out)
        features = self.backbone.feature_extractor_layers(out)
        flat = features.view(features.size(0), -1)

        if (
            not self.projector_cfg.enabled
            or self.physics_projector is None
            or self.spd_metric is None
            or return_raw
        ):
            self._last_projector_penalty = {}
            return flat

        axes, penalties = self.physics_projector(flat)
        self._last_projector_penalty = penalties
        embeddings = self.spd_metric(axes)
        return embeddings

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
            return self.forward_logits(x, data_id=data_id, task_id=task_id)

        embeddings = self.encode(x)
        logits = self.classifier(embeddings)
        payload: Dict[str, Any] = {
            "logits": logits,
            "embeddings": embeddings,
            "projection": embeddings,
            "temperature": self.temperature,
        }
        if self._last_projector_penalty:
            payload["indicator_penalties"] = self._last_projector_penalty
        if self.spd_metric is not None:
            payload["metric_regularizer"] = self.spd_metric.regularization()
        return payload

    # The default classification loss path expects tensors, so expose a dedicated
    # helper that mirrors the original backbone interface when only logits are
    # required.
    def forward_logits(
        self,
        x: torch.Tensor,
        data_id: Optional[Any] = None,
        task_id: Optional[Any] = None,
    ) -> torch.Tensor:
        if (
            not self.projector_cfg.enabled
            or self.physics_projector is None
            or self.spd_metric is None
        ):
            return self.backbone.forward(x, data_id=data_id, task_id=task_id)

        embeddings = self.encode(x)
        logits = self.classifier(embeddings)
        return logits

    # ------------------------------------------------------------------
    # Explainability utilities
    # ------------------------------------------------------------------
    def indicator_penalties(self) -> Dict[str, torch.Tensor]:
        return self._last_projector_penalty

    def explain_indicator_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        if not isinstance(self.classifier, IndicatorClassifier):
            return None
        weights = self.classifier.explain_weights()
        return {key: value.detach().cpu() for key, value in weights.items()}

    def explain_projection_matrix(self) -> Optional[torch.Tensor]:
        if self.physics_projector is None:
            return None
        return self.physics_projector.explain_axes()

    def explain_spd_metric(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.spd_metric is None:
            return None
        state = self.spd_metric.explain()
        return {
            "diag": state.diag,
            "low_rank": state.low_rank,
            "temperature": state.temperature,
        }

    def metric_regularization(self) -> torch.Tensor:
        if self.spd_metric is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return self.spd_metric.regularization()

    def metric_similarity(self, anchors: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        if self.spd_metric is None:
            raise RuntimeError("SPD metric not initialized.")
        return self.spd_metric.similarity(anchors, samples)
