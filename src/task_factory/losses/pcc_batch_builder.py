"""Batch builder utilities for physics-conditioned contrastive loss."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .pcc_loss import PCCBatchContext


@dataclass
class PCCAugmentationConfig:
    enable_phi: bool = True
    time_shift_pct: float = 0.05
    amplitude_range: Tuple[float, float] = (0.8, 1.2)
    stretch_range: Tuple[float, float] = (0.95, 1.05)
    ripple_db: float = 2.0
    views_per_sample: int = 2


@dataclass
class PCCBuilderConfig:
    lambda_pos: float = 1.0
    lambda_neg: float = 1.0
    cos_threshold_deg: float = 25.0
    augmentation: PCCAugmentationConfig = field(default_factory=PCCAugmentationConfig)


class PCCBatchBuilder:
    """Constructs PCC contexts with positives/negatives per query."""

    def __init__(self, config: PCCBuilderConfig, encode_fn) -> None:
        self.config = config
        self.encode_fn = encode_fn
        self._cos_threshold = math.cos(math.radians(config.cos_threshold_deg))

    def build(
        self,
        *,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
        support_embeddings: Optional[torch.Tensor],
        support_labels: Optional[torch.Tensor],
        query_x: torch.Tensor,
        metric,
    ) -> PCCBatchContext:
        device = query_embeddings.device
        phi_embeddings = self._generate_phi_embeddings(query_x) if self.config.augmentation.enable_phi else {}
        positives: List[torch.Tensor] = []
        negatives: List[torch.Tensor] = []
        pos_weights: List[torch.Tensor] = []
        neg_weights: List[torch.Tensor] = []

        prototype_dict = self._build_prototypes(support_embeddings, support_labels)

        for idx, anchor in enumerate(query_embeddings):
            label_val = int(query_labels[idx].item())
            pos_chunks: List[torch.Tensor] = []
            neg_chunks: List[torch.Tensor] = []

            if support_embeddings is not None and support_labels is not None and support_embeddings.size(0) > 0:
                support_mask = support_labels == label_val
                if support_mask.any():
                    pos_chunks.append(support_embeddings[support_mask])
                neg_mask = torch.logical_not(support_mask)
                if neg_mask.any():
                    neg_chunks.append(support_embeddings[neg_mask])

            prototype_vec = prototype_dict.get(label_val)
            if prototype_vec is not None:
                pos_chunks.append(prototype_vec.unsqueeze(0))

            phi_views = phi_embeddings.get(idx)
            if phi_views is not None:
                pos_chunks.append(phi_views)

            query_mask = query_labels != label_val
            if query_mask.any():
                neg_chunks.append(query_embeddings[query_mask])

            if not pos_chunks:
                raise RuntimeError(
                    "PCC builder could not assemble positives; ensure each class has support or augmentations."
                )
            pos_matrix = torch.cat(pos_chunks, dim=0)

            neg_matrix = self._finalise_negatives(neg_chunks, prototype_vec, metric)
            if neg_matrix is None or neg_matrix.numel() == 0:
                raise RuntimeError(
                    "PCC builder could not assemble negatives after filtering; adjust episode sampling."
                )

            positives.append(pos_matrix)
            negatives.append(neg_matrix)
            pos_weights.append(self._positive_weights(anchor.unsqueeze(0), pos_matrix))
            neg_weights.append(self._negative_weights(anchor.unsqueeze(0), neg_matrix))

        return PCCBatchContext(
            query_embeddings=query_embeddings,
            positives=positives,
            negatives=negatives,
            positive_weights=pos_weights,
            negative_weights=neg_weights,
        )

    def _positive_weights(self, anchor: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        residual = torch.mean((anchor - samples) ** 2, dim=-1)
        return torch.exp(-self.config.lambda_pos * residual)

    def _negative_weights(self, anchor: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        residual = torch.mean((anchor - samples) ** 2, dim=-1)
        return torch.exp(self.config.lambda_neg * residual)

    def _finalise_negatives(self, neg_chunks: List[torch.Tensor], prototype_vec, metric) -> Optional[torch.Tensor]:
        if not neg_chunks:
            return None
        negatives = torch.cat(neg_chunks, dim=0)
        if prototype_vec is None:
            return negatives
        proto = prototype_vec.unsqueeze(0)
        sims = metric.similarity(proto, negatives)[0]
        proto_norm = torch.sqrt(metric.similarity(proto, proto)[0, 0].clamp_min(1e-9))
        neg_pair = metric.similarity(negatives, negatives)
        neg_norm = torch.sqrt(torch.diagonal(neg_pair).clamp_min(1e-9))
        cos_values = sims / (proto_norm * neg_norm + 1e-9)
        keep_mask = cos_values <= self._cos_threshold
        if not keep_mask.any():
            return negatives
        return negatives[keep_mask]

    def _build_prototypes(
        self,
        support_embeddings: Optional[torch.Tensor],
        support_labels: Optional[torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        prototypes: Dict[int, torch.Tensor] = {}
        if support_embeddings is None or support_labels is None or support_embeddings.size(0) == 0:
            return prototypes
        for label in support_labels.unique(sorted=True):
            mask = support_labels == label
            if mask.any():
                prototypes[int(label.item())] = support_embeddings[mask].mean(dim=0)
        return prototypes

    def _generate_phi_embeddings(self, query_x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if not self.config.augmentation.enable_phi:
            return {}
        augmented_samples: List[torch.Tensor] = []
        spans: Dict[int, Tuple[int, int]] = {}
        for idx, sample in enumerate(query_x):
            views = self._augment_sample(sample.unsqueeze(0))
            if not views:
                continue
            start = len(augmented_samples)
            augmented_samples.extend(views)
            spans[idx] = (start, start + len(views))
        if not augmented_samples:
            return {}
        stacked = torch.cat(augmented_samples, dim=0)
        embeddings = self.encode_fn(stacked)
        phi_embeddings: Dict[int, torch.Tensor] = {}
        for idx, (start, end) in spans.items():
            phi_embeddings[idx] = embeddings[start:end]
        return phi_embeddings

    def _augment_sample(self, sample: torch.Tensor) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        length = sample.size(-1)
        shift = max(1, int(length * self.config.augmentation.time_shift_pct))
        views.append(torch.roll(sample, shifts=shift, dims=-1))
        views.append(torch.roll(sample, shifts=-shift, dims=-1))

        scale_min, scale_max = self.config.augmentation.amplitude_range
        scale = torch.empty(1, device=sample.device).uniform_(scale_min, scale_max)
        views.append(sample * scale.view(1, 1, 1))

        stretch_min, stretch_max = self.config.augmentation.stretch_range
        stretch = float(torch.empty(1, device=sample.device).uniform_(stretch_min, stretch_max))
        views.append(self._time_stretch(sample, stretch, length))

        ripple = self.config.augmentation.ripple_db / 20.0
        t = torch.linspace(0, math.pi * 2, steps=length, device=sample.device)
        ripple_mask = 1.0 + ripple * torch.sin(t).view(1, 1, -1)
        views.append(sample * ripple_mask)

        return views[: self.config.augmentation.views_per_sample]

    @staticmethod
    def _time_stretch(sample: torch.Tensor, factor: float, target_len: int) -> torch.Tensor:
        stretched = F.interpolate(sample, size=int(target_len * factor), mode="linear", align_corners=False)
        current_len = stretched.size(-1)
        if current_len > target_len:
            stretched = stretched[..., :target_len]
        elif current_len < target_len:
            pad = target_len - current_len
            stretched = F.pad(stretched, (0, pad))
        return stretched
