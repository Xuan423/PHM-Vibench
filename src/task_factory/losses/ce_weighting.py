"""Cross-entropy weighting controllers for Stage 2 contrastive training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

ScaleResult = Tuple[Tensor, Tensor, Dict[str, Tensor]]


def _to_tensor(value: float | Tensor, reference: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return reference.new_tensor(float(value))


class BaseCEWeightController(nn.Module):
    """Base class that scales the CE term and optionally adjusts contrastive loss."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def scale(
        self,
        ce_loss: Tensor,
        *,
        contrastive_loss: Tensor,
        embeddings: Optional[Tensor] = None,
    ) -> ScaleResult:
        raise NotImplementedError


class FixedCEWeightController(BaseCEWeightController):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__("fixed")
        self.register_buffer("weight", torch.tensor(float(weight)))

    def scale(
        self,
        ce_loss: Tensor,
        *,
        contrastive_loss: Tensor,
        embeddings: Optional[Tensor] = None,
    ) -> ScaleResult:
        scaled_ce = ce_loss * self.weight
        metrics = {
            "ce_weight_effective": _to_tensor(self.weight, ce_loss),
        }
        return scaled_ce, contrastive_loss, metrics


class GradAdaptiveCEWeightController(BaseCEWeightController):
    def __init__(
        self,
        init_weight: float = 1.0,
        smoothing: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__("grad-adaptive")
        self.register_buffer("weight", torch.tensor(float(init_weight)))
        self.smoothing = float(smoothing)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)

    def _grad_norm(self, loss: Tensor, embeddings: Optional[Tensor]) -> Optional[Tensor]:
        if embeddings is None or not embeddings.requires_grad:
            return None
        grads = torch.autograd.grad(
            loss,
            embeddings,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = grad.pow(2).sum()
            total = value if total is None else total + value
        if total is None or total <= 0:
            return None
        return torch.sqrt(total + self.eps)

    def scale(
        self,
        ce_loss: Tensor,
        *,
        contrastive_loss: Tensor,
        embeddings: Optional[Tensor] = None,
    ) -> ScaleResult:
        ce_norm = self._grad_norm(ce_loss, embeddings)
        contrastive_norm = self._grad_norm(contrastive_loss, embeddings)
        if ce_norm is not None and contrastive_norm is not None:
            target = (contrastive_norm + self.eps) / (ce_norm + self.eps)
            clamped = torch.clamp(target.detach(), self.min_weight, self.max_weight)
            updated = (1.0 - self.smoothing) * self.weight + self.smoothing * clamped
            self.weight.copy_(updated)
        scaled_ce = ce_loss * self.weight
        metrics = {
            "ce_weight_effective": _to_tensor(self.weight, ce_loss),
        }
        if ce_norm is not None:
            metrics["ce_grad_norm"] = ce_norm.detach()
        if contrastive_norm is not None:
            metrics["contrastive_grad_norm"] = contrastive_norm.detach()
        return scaled_ce, contrastive_loss, metrics


class UncertaintyCEWeightController(BaseCEWeightController):
    def __init__(
        self,
        init_log_sigma_ce: float = 0.0,
        init_log_sigma_contrastive: float = -2.0,
    ) -> None:
        super().__init__("uncertainty")
        self.log_sigma_ce = nn.Parameter(torch.tensor(float(init_log_sigma_ce)))
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(float(init_log_sigma_contrastive)))

    def scale(
        self,
        ce_loss: Tensor,
        *,
        contrastive_loss: Tensor,
        embeddings: Optional[Tensor] = None,
    ) -> ScaleResult:
        ce_factor = torch.exp(-2.0 * self.log_sigma_ce)
        contrastive_factor = torch.exp(-2.0 * self.log_sigma_contrastive)

        scaled_ce = ce_loss * ce_factor + self.log_sigma_ce
        scaled_contrastive = contrastive_loss * contrastive_factor + self.log_sigma_contrastive
        metrics = {
            "ce_weight_effective": ce_factor.detach(),
            "ce_log_sigma": self.log_sigma_ce.detach(),
            "contrastive_weight_effective": contrastive_factor.detach(),
            "contrastive_log_sigma": self.log_sigma_contrastive.detach(),
        }
        return scaled_ce, scaled_contrastive, metrics


def build_ce_weight_controller(cfg) -> BaseCEWeightController:
    """Factory that returns the proper controller given a ce_weighting config block."""

    def _extract(source, key, default=None):
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    ce_cfg = cfg
    mode = str(_extract(ce_cfg, "mode", "fixed")).strip().lower()

    if mode == "grad-adaptive":
        grad_cfg = _extract(ce_cfg, "grad_adaptive", None)
        return GradAdaptiveCEWeightController(
            init_weight=float(_extract(grad_cfg, "init_weight", 1.0)),
            smoothing=float(_extract(grad_cfg, "smoothing", 0.1)),
            min_weight=float(_extract(grad_cfg, "min_weight", 0.1)),
            max_weight=float(_extract(grad_cfg, "max_weight", 10.0)),
        )
    if mode == "uncertainty":
        unc_cfg = _extract(ce_cfg, "uncertainty", None)
        init_log_sigma_contrastive = _extract(
            unc_cfg,
            "init_log_sigma_contrastive",
            _extract(unc_cfg, "init_log_sigma_pcc", -2.0),
        )
        return UncertaintyCEWeightController(
            init_log_sigma_ce=float(_extract(unc_cfg, "init_log_sigma_ce", 0.0)),
            init_log_sigma_contrastive=float(init_log_sigma_contrastive),
        )

    fixed_cfg = _extract(ce_cfg, "fixed", None)
    return FixedCEWeightController(weight=float(_extract(fixed_cfg, "weight", 1.0)))
