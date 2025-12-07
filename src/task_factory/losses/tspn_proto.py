"""Loss utilities for the Transparent Statistical Prototypical Network."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.model_factory.modules import PrototypeMemory
from src.model_factory.X_model.TSPNContrastive import LossWeights


def compute_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    u: torch.Tensor,
    domains: Optional[torch.Tensor],
    prototype_memory: PrototypeMemory,
    projector_module,
    loss_cfg: LossWeights,
) -> Dict[str, torch.Tensor]:
    """Compose CE + ProtoNCE + prototype variance + orthogonal penalties."""
    metrics: Dict[str, torch.Tensor] = {}
    ce = F.cross_entropy(logits, labels)
    metrics["ce"] = ce

    proto_loss = torch.tensor(0.0, device=logits.device)
    proto_valid = torch.tensor(0.0, device=logits.device)
    var_loss = torch.tensor(0.0, device=logits.device)

    prototype_memory.update(u.detach(), labels.detach(), domains.detach() if domains is not None else None)
    sims, targets, mask = prototype_memory.build_proto_logits(
        u,
        labels,
        domains,
        temperature=loss_cfg.temperature,
    )
    if mask.any():
        per_sample = F.cross_entropy(sims, targets, reduction="none")
        proto_loss = (per_sample * mask.float()).sum() / mask.float().sum()
        proto_valid = mask.float().mean()
    metrics["proto_valid"] = proto_valid

    if prototype_memory.config.num_domains is not None:
        domain_proto = prototype_memory.prototypes
        class_mean = prototype_memory.prototypes_mean()
        var_loss = (domain_proto - class_mean.unsqueeze(1)).pow(2).mean()

    metrics["proto"] = proto_loss
    metrics["var"] = var_loss

    ortho = projector_module.ortho_loss()
    metrics["ortho"] = ortho
    phys = torch.tensor(0.0, device=logits.device)
    metrics["phys"] = phys

    total = ce
    total = total + loss_cfg.lambda_proto * proto_loss
    total = total + loss_cfg.lambda_var * var_loss
    total = total + loss_cfg.lambda_ortho * ortho
    total = total + loss_cfg.lambda_phys * phys

    metrics["total"] = total
    return metrics
