"""Stability-based top-K selector for TSPN-CL.

Scores each feature dimension by class separability vs. domain shift,
tracks scores with EMA, and freezes the top-K set after warmup.
"""
from typing import Optional, Tuple

import torch


class StabilityTopKSelector:
    def __init__(
        self,
        top_k: int,
        score_mode: str = "fisher_over_domain_var",
        ema_momentum: float = 0.9,
        warmup_epochs: int = 1,
        eps: float = 1e-6,
    ) -> None:
        self.top_k = top_k
        self.score_mode = score_mode
        self.ema_momentum = ema_momentum
        self.warmup_epochs = warmup_epochs
        self.eps = eps

        self.running_score: Optional[torch.Tensor] = None
        self.frozen_idx: Optional[torch.Tensor] = None
        self.frozen: bool = False

    def _compute_scores(
        self, h_raw: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-dimension stability scores."""
        device = h_raw.device
        K = h_raw.shape[1]
        # Class separability: variance of class means (Fisher-like, weighted by counts)
        uniq_classes = labels.unique()
        class_means = []
        class_weights = []
        for c in uniq_classes:
            mask = labels == c
            class_means.append(h_raw[mask].mean(dim=0))
            class_weights.append(mask.float().mean())  # weight by prevalence
        class_means = torch.stack(class_means, dim=0)
        weights = torch.stack(class_weights, dim=0)
        global_mean = (class_means * weights.unsqueeze(1)).sum(dim=0)
        class_sep = ((class_means - global_mean) ** 2) * weights.unsqueeze(1)
        class_sep = class_sep.sum(dim=0)  # [K]

        # Domain shift: variance of domain means
        uniq_domains = domains.unique()
        domain_means = []
        for d in uniq_domains:
            dmask = domains == d
            domain_means.append(h_raw[dmask].mean(dim=0))
        domain_means = torch.stack(domain_means, dim=0)
        domain_var = domain_means.var(dim=0, unbiased=False)  # [K]

        if self.score_mode == "fisher_over_domain_var":
            score = class_sep / (domain_var + self.eps)
        else:
            score = class_sep / (domain_var + self.eps)

        if self.running_score is None or self.running_score.shape[0] != K:
            self.running_score = score.detach().to(device)
        else:
            m = self.ema_momentum
            self.running_score = m * self.running_score.to(device) + (1 - m) * score.detach().to(device)
        return self.running_score

    def select(
        self,
        h_raw: torch.Tensor,
        labels: torch.Tensor,
        domains: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected features and indices.

        Freezes indices after warmup_epochs; ties broken by original order.
        """
        scores = self._compute_scores(h_raw, labels, domains)

        if (not self.frozen) and (epoch is not None) and (epoch >= self.warmup_epochs):
            # tie-breaker by original order to make deterministic
            tie_breaker = scores + 1e-12 * torch.arange(scores.numel(), device=scores.device)
            topk_idx = torch.topk(tie_breaker, k=min(self.top_k, scores.numel()), dim=0).indices
            self.frozen_idx = topk_idx.detach()
            self.frozen = True

        if self.frozen and self.frozen_idx is not None:
            idx = self.frozen_idx.to(h_raw.device)
        else:
            tie_breaker = scores + 1e-12 * torch.arange(scores.numel(), device=scores.device)
            idx = torch.topk(tie_breaker, k=min(self.top_k, scores.numel()), dim=0).indices
            idx = idx.to(h_raw.device)

        h_sel = h_raw[:, idx]
        return h_sel, idx
