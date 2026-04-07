from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def summarize_prototype_usage(
    labels: torch.Tensor,
    target_proto_probs: torch.Tensor,
    num_classes: int,
    num_prototypes_per_class: int,
) -> torch.Tensor:
    usage = target_proto_probs.new_zeros(num_classes, num_prototypes_per_class)
    counts = target_proto_probs.new_zeros(num_classes)
    for class_id in labels.unique(sorted=True).tolist():
        class_id = int(class_id)
        class_mask = labels == class_id
        if class_mask.sum() == 0:
            continue
        usage[class_id] = target_proto_probs[class_mask].mean(dim=0)
        counts[class_id] = float(class_mask.sum().item())
    return usage


def compute_minimal_prototype_contrastive(
    proto_scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    assignment_temperature: float,
    neg_topk: int = 0,
    positive_mode: str = "logsumexp",
    specialization_margin: float = 0.0,
    specialization_temperature: float | None = None,
    domains: torch.Tensor | None = None,
    target_proto_scores: torch.Tensor | None = None,
    target_proto_probs: torch.Tensor | None = None,
    target_class_neff: torch.Tensor | None = None,
    adaptive_neff_min_scale: float = 1.0,
    adaptive_neff_max_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if proto_scores.ndim != 3:
        raise ValueError(f"Expected proto_scores [B, N, K], got {tuple(proto_scores.shape)}")
    labels = labels.to(proto_scores.device).long()
    tau = max(float(temperature), 1e-6)
    assign_tau = max(float(assignment_temperature), 1e-6)
    specialization_tau = max(
        float(assign_tau if specialization_temperature is None else specialization_temperature),
        1e-6,
    )
    batch_size, num_classes, num_prototypes = proto_scores.shape

    if target_proto_scores is None:
        target_scores = proto_scores[torch.arange(batch_size, device=proto_scores.device), labels]
    else:
        target_scores = target_proto_scores.to(proto_scores.device)
    if target_proto_probs is None:
        target_proto_probs = torch.softmax(target_scores / assign_tau, dim=-1)
    else:
        target_proto_probs = target_proto_probs.to(proto_scores.device)
    if positive_mode == "softmax_expectation":
        positive_scores = (target_proto_probs.detach() * target_scores).sum(dim=-1)
    elif positive_mode == "assigned":
        positive_indices = target_proto_probs.detach().argmax(dim=-1, keepdim=True)
        positive_scores = target_scores.gather(1, positive_indices).squeeze(1)
    else:
        positive_scores = torch.logsumexp(target_scores, dim=-1)
    class_mask = F.one_hot(labels, num_classes=num_classes).to(device=proto_scores.device, dtype=torch.bool)
    negative_scores = proto_scores.masked_select(~class_mask.unsqueeze(-1)).view(batch_size, -1)
    if neg_topk > 0 and negative_scores.shape[1] > neg_topk:
        negative_scores = torch.topk(negative_scores, k=neg_topk, dim=1).values
    if negative_scores.shape[1] == 0:
        negative_scores = proto_scores.new_zeros(batch_size, 1)
    logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1) / tau
    sample_losses = F.cross_entropy(logits, logits.new_zeros(batch_size, dtype=torch.long), reduction="none")
    if target_class_neff is not None:
        neff = target_class_neff.to(proto_scores.device, dtype=sample_losses.dtype)
        sample_scales = (neff / float(max(num_prototypes, 1))).clamp(
            min=float(adaptive_neff_min_scale),
            max=float(adaptive_neff_max_scale),
        )
        total = (sample_losses * sample_scales).mean()
    else:
        sample_scales = proto_scores.new_ones(batch_size)
        total = sample_losses.mean()
    neg_summary = negative_scores

    prototype_usage = summarize_prototype_usage(labels, target_proto_probs, num_classes, num_prototypes)
    uniform = proto_scores.new_full((num_prototypes,), 1.0 / max(num_prototypes, 1))
    active_mask = prototype_usage.sum(dim=1) > 0
    if active_mask.any():
        balance_penalty = ((prototype_usage[active_mask] - uniform) ** 2).mean()
        entropy = -(prototype_usage[active_mask] * prototype_usage[active_mask].clamp_min(1e-6).log()).sum(dim=1)
        n_eff = torch.exp(entropy)
        assignment_ratio = prototype_usage[active_mask].max(dim=1).values.mean()
        occupancy_penalty = (1.0 - n_eff / float(max(num_prototypes, 1))).clamp_min(0.0).pow(2).mean()
    else:
        balance_penalty = proto_scores.new_zeros(())
        n_eff = proto_scores.new_zeros(1)
        assignment_ratio = proto_scores.new_zeros(())
        occupancy_penalty = proto_scores.new_zeros(())

    specialization_penalty = proto_scores.new_zeros(())
    specialization_gap = proto_scores.new_zeros(())
    specialization_competitor_score = proto_scores.new_zeros(())
    if num_prototypes > 1:
        assigned_indices = target_proto_probs.detach().argmax(dim=-1)
        assigned_scores = target_scores.gather(1, assigned_indices.unsqueeze(1)).squeeze(1)
        competitor_mask = ~F.one_hot(
            assigned_indices,
            num_classes=num_prototypes,
        ).to(device=target_scores.device, dtype=torch.bool)
        competitor_scores = target_scores.masked_fill(~competitor_mask, float("-inf"))
        competitor_max = competitor_scores.max(dim=-1).values
        valid_competitor_mask = torch.isfinite(competitor_max)
        if valid_competitor_mask.any():
            assigned_scores_valid = assigned_scores[valid_competitor_mask]
            competitor_scores_valid = competitor_scores[valid_competitor_mask]
            competitor_aggregate = (
                torch.logsumexp(competitor_scores_valid / specialization_tau, dim=-1) * specialization_tau
            )
            competitor_max_valid = competitor_max[valid_competitor_mask]
            specialization_penalty = F.relu(
                competitor_aggregate - assigned_scores_valid + float(specialization_margin)
            ).mean()
            specialization_gap = (assigned_scores_valid - competitor_max_valid).mean()
            specialization_competitor_score = competitor_aggregate.mean()

    return {
        "total": total,
        "proto_nce": total,
        "prototype_contrastive": total,
        "target_proto_probs": target_proto_probs,
        "proto_positive_scores": positive_scores,
        "proto_negative_scores": neg_summary,
        "prototype_usage": prototype_usage,
        "balance_penalty": balance_penalty,
        "occupancy_penalty": occupancy_penalty,
        "assignment_ratio": assignment_ratio,
        "n_eff": n_eff,
        "specialization_penalty": specialization_penalty,
        "specialization_gap": specialization_gap,
        "specialization_competitor_score": specialization_competitor_score,
        "contrastive_sample_scale": sample_scales.mean(),
        "readability_penalty": proto_scores.new_zeros(()),
        "complementarity_penalty": proto_scores.new_zeros(()),
    }
