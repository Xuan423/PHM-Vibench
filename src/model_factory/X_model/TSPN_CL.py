"""TSPN-CL: TSPN backbone with stability Top-K mask, metric head, and Stage2 imprint hooks.

This variant is the experimentation entry for concept/metric learning and few-shot
adaptation. Deprecated "physical concept head" modes are removed; Top-K selection
defaults to fisher score and is applied as a binary mask (no feature reordering).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TSPN import Model as TSPNBackbone
from .utils.topk_selector import StabilityTopKSelector


class Model(nn.Module):
    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.args = args
        self.metadata = metadata

        # Backbone (reuses TSPN layers; classifier inside backbone is not used here)
        self.backbone = TSPNBackbone(args, metadata)
        feature_dim = getattr(self.backbone, "channel_for_classifier", None)
        if feature_dim is None:
            raise ValueError("Backbone missing channel_for_classifier to define feature dimension.")
        self.feature_dim = int(feature_dim)

        # Top-K selection: score-based by default (fisher_over_domain_var) and applied as mask.
        self.top_k = int(getattr(args, "top_k_features", self.feature_dim))
        self.top_k = max(1, min(self.top_k, self.feature_dim))
        self.score_mode = getattr(args, "topk_score_mode", "fisher_over_domain_var")
        self.topk_ema_momentum = getattr(args, "topk_ema_momentum", 0.9)
        self.topk_warmup_epochs = getattr(args, "topk_warmup_epochs", 1)
        self.topk_selector = StabilityTopKSelector(
            top_k=self.top_k,
            score_mode=self.score_mode,
            ema_momentum=self.topk_ema_momentum,
            warmup_epochs=self.topk_warmup_epochs,
        )

        # Metric head: c = V h_masked (V is the factor of a PSD metric M = V^T V).
        self.concept_dim = int(getattr(args, "concept_dim", 32))
        self.metric = nn.Linear(self.feature_dim, self.concept_dim, bias=False)

        # Classifier head(s): linear by default to support Stage2 weight imprinting.
        self.heads = self._build_heads(args.num_classes, self.concept_dim)

        # Contrastive schedule (linear only)
        self.use_contrastive = bool(getattr(args, "use_contrastive_head", False))
        self.lambda_cl_start = float(getattr(args, "lambda_cl_start", 0.0))
        self.lambda_cl_end = float(getattr(args, "lambda_cl_end", 0.1))
        self.temperature = float(getattr(args, "temperature", 0.07))

        # Optional regularizers
        self.sparsity_enable = bool(getattr(args, "sparsity_enable", False))
        self.sparsity_start_ratio = float(getattr(args, "sparsity_start_ratio", 0.7))
        self.sparsity_coeff_row = float(getattr(args, "sparsity_coeff_row", 0.0))
        self.sparsity_coeff_col = float(getattr(args, "sparsity_coeff_col", 0.0))

        # Stage2 optional temperature (stored on model; does not create new classifier)
        self.logit_temperature = float(getattr(args, "stage2_temperature_init", 1.0))

    def _build_heads(self, num_classes: Any, in_dim: int) -> nn.ModuleDict:
        if isinstance(num_classes, dict):
            return nn.ModuleDict({str(k): nn.Linear(in_dim, int(v)) for k, v in num_classes.items()})
        return nn.ModuleDict({"default": nn.Linear(in_dim, int(num_classes))})

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone up to feature extractor, returning h_raw (B x D)."""
        h = x
        for layer in self.backbone.signal_processing_layers:
            h = layer(h)
        h = self.backbone.feature_extractor_layers(h)
        return h

    def _to_tensor_ids(self, file_ids_raw: Any, device: torch.device) -> torch.Tensor:
        if file_ids_raw is None:
            return torch.zeros(1, device=device, dtype=torch.long)
        if torch.is_tensor(file_ids_raw):
            return file_ids_raw.to(device).long()
        try:
            return torch.tensor(file_ids_raw, device=device, dtype=torch.long)
        except Exception:
            return torch.zeros(1, device=device, dtype=torch.long)

    def _get_domains(self, file_ids: torch.Tensor) -> torch.Tensor:
        if self.metadata is None:
            return torch.zeros_like(file_ids, dtype=torch.long)
        dom_list = []
        for fid in file_ids.view(-1).tolist():
            try:
                dom = self.metadata[fid].get("Domain_id", 0)
            except Exception:
                dom = 0
            dom_list.append(int(dom))
        return torch.tensor(dom_list, device=file_ids.device, dtype=torch.long)

    def _pick_head(self, file_ids: Optional[torch.Tensor]) -> nn.Module:
        if len(self.heads) == 1:
            return next(iter(self.heads.values()))
        dataset_id = None
        if file_ids is not None and self.metadata is not None:
            fid0 = int(file_ids.view(-1)[0].item())
            try:
                dataset_id = str(self.metadata[fid0].get("Dataset_id"))
            except Exception:
                dataset_id = None
        if dataset_id and dataset_id in self.heads:
            return self.heads[dataset_id]
        return self.heads[sorted(self.heads.keys())[0]]

    def _topk_mask(self, idx: torch.Tensor, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(dim, device=device, dtype=dtype)
        mask[idx] = 1.0
        return mask

    def encode_with_batch(
        self, batch: Dict[str, Any], epoch: int = 0, *, update_topk: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Return masked physical features and metric embeddings."""
        x = batch["x"]
        labels = batch.get("y")
        file_ids_raw = batch.get("_file_ids_raw", batch.get("file_id"))
        file_ids_tensor = self._to_tensor_ids(file_ids_raw, device=x.device)
        domains = self._get_domains(file_ids_tensor)

        h_raw = self._extract_features(x)

        if labels is None or not update_topk:
            idx = self.topk_selector.get_indices(num_features=h_raw.shape[1], device=h_raw.device)
        else:
            idx = self.topk_selector.select_indices(h_raw, labels.to(h_raw.device), domains, epoch)

        mask = self._topk_mask(idx, dim=h_raw.shape[1], dtype=h_raw.dtype, device=h_raw.device)
        h_masked = h_raw * mask
        c = self.metric(h_masked)

        return {
            "c": c,
            "h_raw": h_raw,
            "h_masked": h_masked,
            "topk_idx": idx,
            "topk_mask": mask,
            "domains": domains,
            "labels": labels,
            "file_ids": file_ids_tensor,
        }

    def forward_with_batch(self, batch: Dict[str, Any], epoch: int = 0) -> Dict[str, torch.Tensor]:
        extras = self.encode_with_batch(batch, epoch=epoch)
        head = self._pick_head(extras.get("file_ids"))
        logits = head(extras["c"]) / max(self.logit_temperature, 1e-6)
        extras["logits"] = logits
        extras["epoch"] = torch.tensor(int(epoch), device=logits.device)
        return extras

    def forward(self, x: torch.Tensor, file_id=None, task_id=None):
        """Legacy forward: returns logits only (uses the first head)."""
        h_raw = self._extract_features(x)
        idx = self.topk_selector.get_indices(num_features=h_raw.shape[1], device=h_raw.device)
        mask = self._topk_mask(idx, dim=h_raw.shape[1], dtype=h_raw.dtype, device=h_raw.device)
        c = self.metric(h_raw * mask)
        head = self._pick_head(None)
        return head(c) / max(self.logit_temperature, 1e-6)

    # --- Stage1 losses (contrastive + optional sparsity) ---

    def compute_contrastive_loss(
        self,
        extras: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        c = extras["c"]
        labels = labels.to(c.device)

        proto_labels, prototypes = self._compute_prototypes(c, labels)
        L_info = self._info_nce(c, prototypes, labels, proto_labels)

        sparsity_pen = self._sparsity_penalty(epoch, total_epochs)
        lambda_w = self._scheduled_lambda(epoch, total_epochs)

        L_phys = torch.tensor(0.0, device=c.device, dtype=c.dtype)
        total = L_info + sparsity_pen
        return total, L_info, L_phys, sparsity_pen, lambda_w

    def _compute_prototypes(self, c: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uniq = labels.unique()
        protos = []
        proto_labels = []
        for lbl in uniq:
            mask = labels == lbl
            protos.append(c[mask].mean(dim=0))
            proto_labels.append(lbl)
        return torch.stack(proto_labels).to(c.device), torch.stack(protos)

    def _info_nce(
        self,
        c: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
        proto_labels: torch.Tensor,
    ) -> torch.Tensor:
        sim = F.cosine_similarity(c.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)  # [B, P]
        logits = sim / max(self.temperature, 1e-6)
        label_to_idx = {int(lbl.item()): idx for idx, lbl in enumerate(proto_labels)}
        target = torch.tensor([label_to_idx[int(l.item())] for l in labels], device=c.device)
        return F.cross_entropy(logits, target)

    def _scheduled_lambda(self, epoch: Optional[int], total_epochs: Optional[int]) -> float:
        if epoch is None or total_epochs is None or total_epochs <= 0:
            return float(self.lambda_cl_end)
        ratio = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
        return float(self.lambda_cl_start + (self.lambda_cl_end - self.lambda_cl_start) * ratio)

    def _sparsity_penalty(self, epoch: Optional[int], total_epochs: Optional[int]) -> torch.Tensor:
        if not self.sparsity_enable:
            return torch.tensor(0.0, device=self.metric.weight.device, dtype=self.metric.weight.dtype)
        if epoch is None or total_epochs is None or total_epochs <= 0:
            return torch.tensor(0.0, device=self.metric.weight.device, dtype=self.metric.weight.dtype)
        if (epoch / max(total_epochs, 1)) < self.sparsity_start_ratio:
            return torch.tensor(0.0, device=self.metric.weight.device, dtype=self.metric.weight.dtype)

        V = self.metric.weight  # [R, D]
        row_pen = torch.norm(V, p=1, dim=1).mean() if self.sparsity_coeff_row > 0 else torch.tensor(0.0, device=V.device, dtype=V.dtype)
        col_pen = torch.norm(V, p=2, dim=0).mean() if self.sparsity_coeff_col > 0 else torch.tensor(0.0, device=V.device, dtype=V.dtype)
        return self.sparsity_coeff_row * row_pen + self.sparsity_coeff_col * col_pen

    # --- Stage2 helpers ---

    def get_active_linear_head(self, file_ids: Optional[torch.Tensor]) -> nn.Linear:
        """Return the active classifier head as an nn.Linear (for Stage2 imprint).

        This returns the same module instance used during forward. No new modules
        are created.
        """
        head = self._pick_head(file_ids)
        if isinstance(head, nn.Linear):
            return head
        # Allow Sequential but require last layer to be Linear.
        if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[-1], nn.Linear):
            return head[-1]
        raise TypeError(f"Unsupported head type for imprinting: {type(head)}")
