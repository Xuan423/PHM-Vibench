from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CooperativePrototypeFusion(nn.Module):
    """Fuse branch prototype predictions through an interpretable evidence posterior."""

    def __init__(
        self,
        weight_floor: float = 0.15,
        max_share: float = 0.65,
        evidence_mix: float = 0.5,
        confidence_mix: float = 0.15,
        agreement_mix: float = 0.35,
        logit_scale_min: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight_floor = float(weight_floor)
        self.max_share = float(max_share)
        self.evidence_mix = float(evidence_mix)
        self.confidence_mix = float(confidence_mix)
        self.agreement_mix = float(agreement_mix)
        self.logit_scale_min = float(logit_scale_min)

    @staticmethod
    def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return x / x.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _confidence(probs: torch.Tensor) -> torch.Tensor:
        class_count = max(int(probs.shape[-1]), 2)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(float(class_count))
        return confidence.clamp(0.0, 1.0)

    @staticmethod
    def _agreement(probs: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        anchor_norm = F.normalize(anchor, dim=-1)
        probs_norm = F.normalize(probs, dim=-1)
        cosine = (probs_norm * anchor_norm.unsqueeze(1)).sum(dim=-1)
        return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    def _cap_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.max_share >= 1.0 or weights.shape[1] <= 1:
            return self._normalize_rows(weights)
        capped = weights.clamp(max=self.max_share)
        remainder = (1.0 - capped.sum(dim=1, keepdim=True)).clamp_min(0.0)
        under_mask = capped < self.max_share
        under_mass = (capped * under_mask).sum(dim=1, keepdim=True)
        redistribute = torch.where(
            under_mass > 1e-6,
            remainder * capped * under_mask / under_mass.clamp_min(1e-6),
            remainder / float(weights.shape[1]),
        )
        fused = capped + redistribute
        return self._normalize_rows(fused)

    def _logit_scales(
        self,
        evidence: torch.Tensor,
        agreement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reliability = self._normalize_rows(torch.sqrt((evidence * agreement).clamp_min(1e-6)))
        if self.logit_scale_min >= 0.999 or reliability.shape[1] <= 1:
            return torch.ones_like(reliability), reliability
        uniform = 1.0 / float(reliability.shape[1])
        relative = ((reliability - uniform) / max(1.0 - uniform, 1e-6)).clamp(0.0, 1.0)
        scales = self.logit_scale_min + (1.0 - self.logit_scale_min) * relative
        return scales, reliability

    def _posterior_weights(
        self,
        evidence: torch.Tensor,
        confidence: torch.Tensor,
        agreement: torch.Tensor,
    ) -> torch.Tensor:
        mix_total = max(
            self.evidence_mix + self.confidence_mix + self.agreement_mix,
            1e-6,
        )
        mix_e = self.evidence_mix / mix_total
        mix_c = self.confidence_mix / mix_total
        mix_a = self.agreement_mix / mix_total
        posterior_logits = (
            mix_e * evidence.clamp_min(1e-6).log()
            + mix_c * confidence.clamp_min(1e-6).log()
            + mix_a * agreement.clamp_min(1e-6).log()
        )
        posterior = torch.softmax(posterior_logits, dim=1)
        if self.weight_floor > 0.0:
            posterior = (1.0 - self.weight_floor) * posterior + self.weight_floor * evidence
        return self._cap_weights(posterior)

    def forward(
        self,
        evidence_weights: torch.Tensor,
        head_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if evidence_weights.ndim != 2:
            raise ValueError(f"Expected evidence_weights [B, H], got {tuple(evidence_weights.shape)}.")
        if head_logits.ndim != 3:
            raise ValueError(f"Expected head_logits [B, H, C], got {tuple(head_logits.shape)}.")
        if evidence_weights.shape[:1] != head_logits.shape[:1] or evidence_weights.shape[1] != head_logits.shape[1]:
            raise ValueError(
                f"Evidence/logit branch mismatch: {tuple(evidence_weights.shape)} vs {tuple(head_logits.shape)}."
            )

        evidence = self._normalize_rows(evidence_weights)
        raw_probs = torch.softmax(head_logits, dim=-1)
        confidence = self._normalize_rows(self._confidence(raw_probs))
        joint_anchor_probs = raw_probs[:, 2]
        agreement = self._normalize_rows(self._agreement(raw_probs, joint_anchor_probs))
        logit_scales, reliability = self._logit_scales(evidence, agreement)
        calibrated_logits = head_logits * logit_scales.unsqueeze(-1)
        calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
        fused_weights = self._posterior_weights(
            evidence=evidence,
            confidence=confidence,
            agreement=agreement,
        )
        fused_logits = torch.sum(
            fused_weights.unsqueeze(-1) * calibrated_logits,
            dim=1,
        )
        fused_probs = torch.softmax(fused_logits, dim=-1)

        return {
            "fused_logits": fused_logits,
            "fused_probs": fused_probs,
            "calibrated_head_logits": calibrated_logits,
            "weights": fused_weights,
            "evidence_weights": evidence,
            "confidence_weights": confidence,
            "agreement_weights": agreement,
            "reliability_weights": reliability,
            "head_logit_scales": logit_scales,
            "head_confidence": self._confidence(calibrated_probs),
            "head_agreement": self._agreement(calibrated_probs, calibrated_probs[:, 2]),
        }
