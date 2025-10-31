"""Contrastive wrapper around the baseline TSPN model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .TSPN import Model as TSPNModel


@dataclass
class ProjectorPriorConfig:
    path: Optional[str] = None
    format: str = "npy"
    temperature: float = 1.0


@dataclass
class PhysicalProjectorConfig:
    enabled: bool = False
    indicator_dim: int = 0
    init: str = "kaiming"
    sparsity_l1: float = 0.0
    sparsity_group: float = 0.0
    max_active: Optional[int] = None
    prior: ProjectorPriorConfig = field(default_factory=ProjectorPriorConfig)


@dataclass
class ContrastiveConfig:
    enabled: bool
    projection_hidden: int
    projection_dim: int
    temperature: float
    loss_weight: float
    mode: str = "supervised"
    return_embeddings: bool = True
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

    max_active_raw = getattr(raw_cfg, "max_active", None)
    max_active = int(max_active_raw) if max_active_raw not in (None, False) else None

    return PhysicalProjectorConfig(
        enabled=bool(getattr(raw_cfg, "enabled", False)),
        indicator_dim=int(getattr(raw_cfg, "indicator_dim", 0)),
        init=str(getattr(raw_cfg, "init", "kaiming")),
        sparsity_l1=float(getattr(raw_cfg, "sparsity_l1", 0.0)),
        sparsity_group=float(getattr(raw_cfg, "sparsity_group", 0.0)),
        max_active=max_active,
        prior=prior_cfg,
    )


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
    physical_projector = _to_physical_projector_config(getattr(raw_cfg, "physical_projector", None))

    return ContrastiveConfig(
        enabled=enabled,
        projection_hidden=projection_hidden,
        projection_dim=projection_dim,
        temperature=temperature,
        loss_weight=loss_weight,
        mode=mode,
        return_embeddings=return_embeddings,
        physical_projector=physical_projector,
    )


class PhysicalProjector(nn.Module):
    """Map backbone features into a constrained physical indicator space."""

    def __init__(
        self,
        input_dim: int,
        cfg: PhysicalProjectorConfig,
        *,
        prior_mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if cfg.indicator_dim <= 0:
            raise ValueError("PhysicalProjector enabled but indicator_dim is not positive.")

        self.cfg = cfg
        self.linear = nn.Linear(input_dim, cfg.indicator_dim, bias=True)
        self.register_buffer("prior_mask", prior_mask)

        if cfg.init == "prior" and prior_mask is not None:
            with torch.no_grad():
                self.linear.weight.data.mul_(prior_mask)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        flat = features.view(features.size(0), -1)
        weight = self.linear.weight
        if self.prior_mask is not None:
            weight = weight * self.prior_mask

        indicators = F.linear(flat, weight, self.linear.bias)

        penalties: Dict[str, torch.Tensor] = {}
        if self.cfg.sparsity_l1 > 0:
            penalties["projector_l1"] = self.cfg.sparsity_l1 * weight.abs().mean()
        if self.cfg.sparsity_group > 0:
            penalties["projector_group"] = self.cfg.sparsity_group * torch.norm(weight, dim=1).mean()
        if self.cfg.max_active is not None and self.cfg.max_active > 0 and weight.size(1) > self.cfg.max_active:
            sorted_vals, _ = torch.sort(weight.abs(), dim=1, descending=True)
            surplus = sorted_vals[:, self.cfg.max_active :]
            penalties["projector_max_active"] = surplus.mean()

        return indicators, penalties

    def explain_projection(self) -> torch.Tensor:
        weight = self.linear.weight
        if self.prior_mask is not None:
            weight = weight * self.prior_mask
        return weight.detach().clone()


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
        self.mode = self.contrastive_cfg.mode
        self._last_projector_penalty: Dict[str, torch.Tensor] = {}

        device = getattr(self.args, "device", "cpu")
        self.projector_cfg = self.contrastive_cfg.physical_projector
        self.projector: Optional[PhysicalProjector] = None
        self.classifier: nn.Module = self.backbone.clf
        self._embedding_dim = self.feature_dim

        if self.projector_cfg.enabled:
            self.projector = self._build_physical_projector(device=device)
            num_classes = getattr(self.backbone, "num_classes", None)
            if num_classes is None:
                num_classes = int(getattr(args, "num_classes", 0))
            if num_classes <= 0:
                raise ValueError("Physical projector requires a valid num_classes value.")
            self.classifier = IndicatorClassifier(self.projector_cfg.indicator_dim, num_classes).to(device)
            self._embedding_dim = self.projector_cfg.indicator_dim

        self.projection_head: Optional[nn.Module] = None
        if self.contrastive_cfg.enabled:
            self._validate_contrastive_config()
            self.projection_head = self._build_projection_head(self._embedding_dim, device=device)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _validate_contrastive_config(self) -> None:
        cfg = self.contrastive_cfg
        missing: Dict[str, Any] = {}
        if cfg.projection_dim < 0:
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

    def _build_projection_head(self, input_dim: int, *, device: Optional[Any] = None) -> nn.Module:
        cfg = self.contrastive_cfg
        proj_dim = cfg.projection_dim
        hidden = cfg.projection_hidden

        if proj_dim <= 0 or proj_dim == input_dim:
            return nn.Identity()

        layers = []
        if hidden and hidden > 0:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden, proj_dim))
        else:
            layers.append(nn.Linear(input_dim, proj_dim))

        projector = nn.Sequential(*layers)
        target_device = device if device is not None else getattr(self.args, "device", "cpu")
        return projector.to(target_device)

    def _build_physical_projector(self, *, device: Optional[Any]) -> PhysicalProjector:
        mask = self._load_projector_prior_mask(device=device)
        projector = PhysicalProjector(
            input_dim=self.feature_dim,
            cfg=self.projector_cfg,
            prior_mask=mask,
        )
        return projector.to(device)

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

        expected = (self.projector_cfg.indicator_dim, self.feature_dim)
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

        if not self.projector_cfg.enabled or self.projector is None or return_raw:
            self._last_projector_penalty = {}
            return flat

        indicators, penalties = self.projector(flat)
        self._last_projector_penalty = penalties
        return indicators

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the contrastive projection head if it exists."""
        if self.projection_head is None:
            return embeddings
        flat = embeddings.view(embeddings.size(0), -1)
        return self.projection_head(flat)

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
        projection_input = embeddings
        projected = self.project(projection_input)
        payload: Dict[str, Any] = {
            "logits": logits,
            "embeddings": embeddings,
            "projection": projected,
            "temperature": self.temperature,
        }
        if self._last_projector_penalty:
            payload["indicator_penalties"] = self._last_projector_penalty
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
        if not self.projector_cfg.enabled or self.projector is None:
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
        if self.projector is None:
            return None
        return self.projector.explain_projection().cpu()
