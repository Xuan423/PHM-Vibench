"""Physics-conditioned contrastive loss utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class PCCLossConfig:
    temperature: float = 0.2
    eps: float = 1e-8


@dataclass
class PCCBatchContext:
    """Holds per-query positive/negative sets for PCC."""

    query_embeddings: torch.Tensor
    positives: List[torch.Tensor]
    negatives: List[torch.Tensor]
    positive_weights: List[torch.Tensor]
    negative_weights: List[torch.Tensor]


class PhysicsConditionedContrastiveLoss:
    """Computes the PCC objective given per-query positive/negative sets."""

    def __init__(self, config: PCCLossConfig) -> None:
        self.config = config

    def __call__(
        self,
        metric,
        context: PCCBatchContext,
    ) -> Dict[str, torch.Tensor]:
        device = context.query_embeddings.device
        tau = torch.as_tensor(self.config.temperature, device=device, dtype=context.query_embeddings.dtype)
        eps = torch.as_tensor(self.config.eps, device=device, dtype=context.query_embeddings.dtype)

        losses: List[torch.Tensor] = []
        positive_total = torch.tensor(0.0, device=device, dtype=context.query_embeddings.dtype)
        anchor_total = torch.tensor(0.0, device=device, dtype=context.query_embeddings.dtype)

        for idx, anchor in enumerate(context.query_embeddings):
            pos = context.positives[idx]
            neg = context.negatives[idx]
            w_pos = context.positive_weights[idx]
            w_neg = context.negative_weights[idx]
            if pos is None or pos.numel() == 0:
                raise RuntimeError(f"PCC context missing positives for anchor index {idx}.")
            if neg is None or neg.numel() == 0:
                raise RuntimeError(f"PCC context missing negatives for anchor index {idx}.")

            anchor_view = anchor.unsqueeze(0)
            pos_logits = metric.similarity(anchor_view, pos).squeeze(0) / tau
            neg_logits = metric.similarity(anchor_view, neg).squeeze(0) / tau

            pos_exp = w_pos * torch.exp(pos_logits)
            neg_exp = w_neg * torch.exp(neg_logits)

            numerator = pos_exp.sum()
            denominator = numerator + neg_exp.sum()
            losses.append(-torch.log((numerator + eps) / (denominator + eps)))
            positive_total = positive_total + torch.tensor(float(pos.shape[0]), device=device, dtype=positive_total.dtype)
            anchor_total = anchor_total + 1.0

        if not losses:
            raise RuntimeError("PCC context produced no valid anchors.")

        loss_tensor = torch.stack(losses).mean()
        return {
            "loss": loss_tensor,
            "positive_count": positive_total,
            "anchor_count": anchor_total,
        }
