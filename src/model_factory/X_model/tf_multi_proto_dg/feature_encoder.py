from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_factory.X_model.utils.topk_selector import StabilityTopKSelector


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        concept_dim: int,
        num_classes: int | Dict[str, int],
        metadata: Any,
        use_topk_selector: bool = False,
        top_k_features: Optional[int] = None,
        score_mode: str = "fisher_over_domain_var",
        ema_momentum: float = 0.9,
        warmup_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.concept_dim = int(concept_dim)
        self.metadata = metadata
        self.use_topk_selector = bool(use_topk_selector)

        self.metric = nn.Linear(self.feature_dim, self.concept_dim, bias=False)
        self.heads = self._build_heads(num_classes, self.concept_dim)

        self.topk_selector = None
        if self.use_topk_selector:
            if top_k_features is None:
                raise ValueError("top_k_features must be provided when use_topk_selector is enabled.")
            self.topk_selector = StabilityTopKSelector(
                top_k=min(int(top_k_features), self.feature_dim),
                score_mode=score_mode,
                ema_momentum=ema_momentum,
                warmup_epochs=warmup_epochs,
            )

    @staticmethod
    def _build_heads(num_classes: int | Dict[str, int], in_dim: int) -> nn.ModuleDict:
        if isinstance(num_classes, dict):
            return nn.ModuleDict({str(key): nn.Linear(in_dim, int(value)) for key, value in num_classes.items()})
        return nn.ModuleDict({"default": nn.Linear(in_dim, int(num_classes))})

    def _to_tensor_ids(self, file_ids_raw: Any, device: torch.device) -> torch.Tensor:
        if file_ids_raw is None:
            return torch.zeros(1, device=device, dtype=torch.long)
        if torch.is_tensor(file_ids_raw):
            return file_ids_raw.to(device).long()
        try:
            return torch.as_tensor(file_ids_raw, device=device, dtype=torch.long)
        except Exception:
            return torch.zeros(1, device=device, dtype=torch.long)

    def resolve_head_key(self, file_ids: Optional[torch.Tensor]) -> str:
        if len(self.heads) == 1:
            return next(iter(self.heads.keys()))
        dataset_id = None
        if file_ids is not None and self.metadata is not None and file_ids.numel() > 0:
            try:
                dataset_id = str(self.metadata[int(file_ids.view(-1)[0].item())].get("Dataset_id"))
            except Exception:
                dataset_id = None
        if dataset_id and dataset_id in self.heads:
            return dataset_id
        return sorted(self.heads.keys())[0]

    def get_domains(self, file_ids: torch.Tensor) -> torch.Tensor:
        if self.metadata is None:
            return torch.zeros_like(file_ids, dtype=torch.long)
        domains = []
        for fid in file_ids.view(-1).tolist():
            try:
                domains.append(int(self.metadata[int(fid)].get("Domain_id", 0)))
            except Exception:
                domains.append(0)
        return torch.as_tensor(domains, device=file_ids.device, dtype=torch.long)

    def _topk_mask(self, idx: torch.Tensor, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(dim, device=device, dtype=dtype)
        mask[idx] = 1.0
        return mask

    def encode(
        self,
        h_raw: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor | str | None]:
        file_ids = self._to_tensor_ids(file_ids_raw, h_raw.device)
        domains = self.get_domains(file_ids)

        topk_idx = None
        topk_mask = None
        h_masked = h_raw
        if self.topk_selector is not None:
            if labels is not None:
                topk_idx = self.topk_selector.select_indices(
                    h_raw, labels.to(h_raw.device), domains, epoch
                )
            else:
                topk_idx = self.topk_selector.get_indices(num_features=h_raw.shape[1], device=h_raw.device)
            topk_mask = self._topk_mask(topk_idx, h_raw.shape[1], h_raw.dtype, h_raw.device)
            h_masked = h_raw * topk_mask

        c = F.normalize(self.metric(h_masked), dim=1)
        head_key = self.resolve_head_key(file_ids)
        logits = self.heads[head_key](c)
        return {
            "logits": logits,
            "c": c,
            "h_raw": h_raw,
            "h_masked": h_masked,
            "topk_idx": topk_idx,
            "topk_mask": topk_mask,
            "domains": domains,
            "file_ids": file_ids,
            "head_key": head_key,
        }
