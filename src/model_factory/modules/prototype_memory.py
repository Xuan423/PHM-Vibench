"""Prototype memory for ProtoNCE with optional domain conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PrototypeMemoryConfig:
    num_classes: int
    feat_dim: int
    num_domains: Optional[int] = None
    momentum: float = 0.99


class PrototypeMemory(nn.Module):
    """Momentum-based prototype queue with optional domain dimension."""

    def __init__(self, config: PrototypeMemoryConfig) -> None:
        super().__init__()
        self.config = config
        shape = (
            config.num_classes,
            config.num_domains if config.num_domains is not None else 1,
            config.feat_dim,
        )
        self.register_buffer("prototypes", torch.zeros(shape))
        self.register_buffer("counts", torch.zeros(shape[:-1]))

    def _domain_index(self, domains: Optional[torch.Tensor]) -> torch.Tensor:
        if self.config.num_domains is None:
            return torch.zeros_like(domains if domains is not None else torch.zeros(1, dtype=torch.long))
        if domains is None:
            raise ValueError("Domain-aware prototypes require domain labels in the batch.")
        return domains.clamp(min=0, max=self.config.num_domains - 1)

    def update(self, feats: torch.Tensor, labels: torch.Tensor, domains: Optional[torch.Tensor] = None) -> None:
        """Momentum update using the current batch."""
        with torch.no_grad():
            dom_idx = self._domain_index(domains).to(labels.device)
            labels = labels.to(self.prototypes.device)
            dom_idx = dom_idx.to(self.prototypes.device)
            feats = feats.to(self.prototypes.device)

            for cls in torch.unique(labels):
                mask = labels == cls
                if mask.any():
                    dom_subset = dom_idx[mask]
                    feat_subset = feats[mask]
                    for dom in torch.unique(dom_subset):
                        dom_mask = dom_subset == dom
                        batch_vec = feat_subset[dom_mask].mean(dim=0)
                        c_idx = int(cls.item())
                        d_idx = int(dom.item())
                        proto = self.prototypes[c_idx, d_idx]
                        momentum = self.config.momentum
                        if self.counts[c_idx, d_idx] == 0:
                            proto.copy_(batch_vec)
                        else:
                            proto.mul_(momentum).add_(batch_vec * (1 - momentum))
                        self.counts[c_idx, d_idx] += dom_mask.sum()

    def build_proto_logits(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        domains: Optional[torch.Tensor],
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ProtoNCE logits, targets, and a valid mask."""
        dom_idx = self._domain_index(domains).to(self.prototypes.device)
        if self.config.num_domains is None:
            proto = self.prototypes[:, 0, :]
            counts = self.counts[:, 0]
            target_proto = proto[labels]
            valid_mask = counts[labels] > 0
            sims = feats @ proto.t() / temperature
        else:
            proto = self.prototypes.view(self.config.num_classes * self.config.num_domains, -1)
            counts = self.counts.view(-1)
            offsets = labels * self.config.num_domains + dom_idx
            target_proto = proto[offsets]
            valid_mask = counts[offsets] > 0
            sims = feats @ proto.t() / temperature
            labels = offsets

        targets = torch.argmax(target_proto @ proto.t(), dim=1) if self.config.num_domains else labels
        return sims, targets, valid_mask

    def prototypes_mean(self) -> torch.Tensor:
        """Return per-class mean prototype for variance regularisation."""
        if self.config.num_domains is None:
            return self.prototypes[:, 0, :]
        domain_mean = self.prototypes.mean(dim=1)
        return domain_mean
