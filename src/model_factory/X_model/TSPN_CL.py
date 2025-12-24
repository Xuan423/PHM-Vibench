"""TSPN-CL: TSPN backbone with stability Top-K mask, metric head, and Stage2 imprint hooks.

This variant is the experimentation entry for concept/metric learning and few-shot
adaptation. Deprecated "physical concept head" modes are removed; Top-K selection
defaults to fisher score and is applied as a binary mask (no feature reordering).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

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

        # ---- Multi-prototype ProtoNCE (minimal, hard-only) ----
        self.num_prototypes_per_class = max(1, int(getattr(args, "num_prototypes_per_class", 1)))
        self.proto_neg_k = max(0, int(getattr(args, "proto_neg_k", 0)))
        self.proto_ema_momentum = float(getattr(args, "proto_ema_momentum", 0.95))
        self.proto_ema_momentum = float(min(max(self.proto_ema_momentum, 0.0), 0.999))
        self.proto_empty_reset_steps = 200  # fixed by spec

        # Complementarity (lightweight, prototype-space; no orthogonality)
        self.complementarity_enabled = bool(getattr(args, "complementarity_enabled", False))
        self.complementarity_weight = float(getattr(args, "complementarity_weight", 0.0))

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

        # Optional diagnostics exports (Top-K + sparsity). Off by default.
        self.export_diagnostics = bool(getattr(args, "export_diagnostics", False))

        # PrototypeBank buffers (one bank per classifier head key).
        self._proto_head_key_to_bufkey: Dict[str, str] = {}
        self._register_prototype_banks()
        # Python-only per-epoch member ids (only populated when diagnostics are enabled).
        self._proto_members: Dict[str, list[list[list[int]]]] = {}

    def _build_heads(self, num_classes: Any, in_dim: int) -> nn.ModuleDict:
        if isinstance(num_classes, dict):
            return nn.ModuleDict({str(k): nn.Linear(in_dim, int(v)) for k, v in num_classes.items()})
        return nn.ModuleDict({"default": nn.Linear(in_dim, int(num_classes))})

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in str(key))

    def _register_prototype_banks(self) -> None:
        """Register PrototypeBank buffers for each classifier head key."""
        for head_key, head in self.heads.items():
            bufkey = self._sanitize_key(head_key)
            self._proto_head_key_to_bufkey[head_key] = bufkey

            num_classes = int(getattr(head, "out_features", 0))
            if num_classes <= 0:
                continue

            P = torch.zeros(num_classes, self.num_prototypes_per_class, self.concept_dim)
            self.register_buffer(f"_proto_P_{bufkey}", P)
            self.register_buffer(f"_proto_filled_{bufkey}", torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer(
                f"_proto_since_{bufkey}",
                torch.zeros(num_classes, self.num_prototypes_per_class, dtype=torch.long),
            )
            self.register_buffer(
                f"_proto_epoch_counts_{bufkey}",
                torch.zeros(num_classes, self.num_prototypes_per_class, dtype=torch.long),
            )
            self.register_buffer(f"_proto_epoch_{bufkey}", torch.tensor(-1, dtype=torch.long))

    @staticmethod
    def _deterministic_topk(sim: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k with deterministic tie-break (smaller index wins)."""
        if k <= 0:
            return torch.empty(sim.shape[0], 0, device=sim.device, dtype=torch.long)
        n = int(sim.shape[1])
        indices = torch.arange(n, device=sim.device, dtype=torch.float64)
        tie_breaker = sim.to(torch.float64) - 1e-12 * indices.unsqueeze(0)
        return torch.topk(tie_breaker, k=min(int(k), n), dim=1).indices

    def _resolve_head_key(self, file_ids: Optional[torch.Tensor]) -> str:
        if len(self.heads) == 1:
            return next(iter(self.heads.keys()))
        dataset_id = None
        if file_ids is not None and self.metadata is not None:
            fid0 = int(file_ids.view(-1)[0].item())
            try:
                dataset_id = str(self.metadata[fid0].get("Dataset_id"))
            except Exception:
                dataset_id = None
        if dataset_id and dataset_id in self.heads:
            return dataset_id
        return sorted(self.heads.keys())[0]

    def _get_proto_bank(self, head_key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bufkey = self._proto_head_key_to_bufkey[head_key]
        P = getattr(self, f"_proto_P_{bufkey}")
        filled = getattr(self, f"_proto_filled_{bufkey}")
        since = getattr(self, f"_proto_since_{bufkey}")
        epoch_counts = getattr(self, f"_proto_epoch_counts_{bufkey}")
        epoch_tensor = getattr(self, f"_proto_epoch_{bufkey}")
        return P, filled, since, epoch_counts, epoch_tensor

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
    ) -> Dict[str, Union[torch.Tensor, float]]:
        c = extras["c"]
        labels = labels.to(c.device).long()

        head_key = self._resolve_head_key(extras.get("file_ids"))
        P, filled, since, epoch_counts, epoch_tensor = self._get_proto_bank(head_key)

        # Reset per-epoch counters when epoch changes.
        if epoch is not None and int(epoch_tensor.item()) != int(epoch):
            epoch_counts.zero_()
            epoch_tensor.fill_(int(epoch))
            if self.export_diagnostics:
                self._proto_members[head_key] = [
                    [[] for _ in range(self.num_prototypes_per_class)] for _ in range(P.shape[0])
                ]

        # Normalize embeddings for cosine similarity.
        c_norm = F.normalize(c, p=2, dim=-1)

        # Fixed initialization from first observed embeddings (train-time only).
        if self.training:
            with torch.no_grad():
                for i in range(c_norm.shape[0]):
                    y = int(labels[i].item())
                    if y < 0 or y >= int(P.shape[0]):
                        continue
                    k = int(filled[y].item())
                    if k <= 0:
                        P[y, 0].copy_(c_norm[i].detach())
                        if self.num_prototypes_per_class > 1:
                            P[y, 1:].copy_(P[y, 0].detach().unsqueeze(0))
                        filled[y] = 1
                    elif k < self.num_prototypes_per_class:
                        P[y, k].copy_(c_norm[i].detach())
                        filled[y] = k + 1

        C = int(P.shape[0])
        M = int(P.shape[1])
        num_protos = int(C * M)
        complementarity_pen = torch.tensor(0.0, device=c.device, dtype=c.dtype)
        if C <= 1 or num_protos <= M:
            L_info = torch.tensor(0.0, device=c.device, dtype=c.dtype)
        else:
            P_flat = F.normalize(P.view(num_protos, -1), p=2, dim=-1)
            sim_all = c_norm @ P_flat.t()  # [B, C*M]
            sim_by_class = sim_all.view(sim_all.shape[0], C, M)
            within = sim_by_class[torch.arange(sim_all.shape[0], device=c.device), labels, :]  # [B, M]
            a = within.argmax(dim=1)  # deterministic: first max index
            pos_idx = labels * M + a  # [B]

            tau = max(float(self.temperature), 1e-6)
            if self.proto_neg_k == 0:
                logits = sim_all / tau
                logits_view = logits.view(sim_all.shape[0], C, M)

                # Mask out true-class prototypes except the positive one, without in-place ops on tensors
                # that require grad (avoids autograd versioning errors).
                keep = torch.ones_like(logits_view, dtype=torch.bool)
                batch_idx = torch.arange(sim_all.shape[0], device=c.device)
                keep[batch_idx, labels, :] = False
                keep[batch_idx, labels, a] = True
                logits_masked = logits_view.masked_fill(~keep, -float("inf")).view(sim_all.shape[0], num_protos)
                L_info = F.cross_entropy(logits_masked, pos_idx)
            else:
                num_neg_avail = num_protos - M
                k = min(int(self.proto_neg_k), int(num_neg_avail))
                if k <= 0:
                    L_info = torch.tensor(0.0, device=c.device, dtype=c.dtype)
                else:
                    # Hard-negative selection does not need gradients. Use a detached similarity matrix.
                    sim_detached = sim_all.detach()
                    sim_detached_view = sim_detached.view(sim_all.shape[0], C, M)
                    batch_idx = torch.arange(sim_all.shape[0], device=c.device)
                    exclude = torch.zeros_like(sim_detached_view, dtype=torch.bool)
                    exclude[batch_idx, labels, :] = True
                    sim_masked = sim_detached_view.masked_fill(exclude, -float("inf")).view(sim_all.shape[0], num_protos)
                    topk_idx = self._deterministic_topk(sim_masked, k=k)  # [B, k]

                    s_pos = sim_all.gather(1, pos_idx.unsqueeze(1)).squeeze(1) / tau
                    s_neg = sim_all.gather(1, topk_idx) / tau
                    scores = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
                    target = torch.zeros(scores.shape[0], device=c.device, dtype=torch.long)
                    L_info = F.cross_entropy(scores, target)

            # PrototypeBank updates (train only, every step).
            if self.training:
                with torch.no_grad():
                    # Update occupancy + since_assigned.
                    since.add_(1)

                    # Track hard member ids for diagnostics/debugging only.
                    if self.export_diagnostics and "file_ids" in extras and head_key in self._proto_members:
                        ids = extras["file_ids"].detach().cpu().view(-1).tolist()
                        ys = labels.detach().cpu().view(-1).tolist()
                        ms = a.detach().cpu().view(-1).tolist()
                        members = self._proto_members[head_key]
                        for fid, y, m in zip(ids, ys, ms):
                            if 0 <= y < len(members) and 0 <= m < len(members[y]):
                                members[y][m].append(int(fid))

                    # EMA update per prototype using mean assigned embedding.
                    flat_idx = (labels * M + a).detach()
                    sums = torch.zeros(num_protos, c_norm.shape[1], device=c.device, dtype=c_norm.dtype)
                    sums.index_add_(0, flat_idx, c_norm.detach())
                    counts_long = torch.bincount(flat_idx, minlength=num_protos)
                    counts = counts_long.to(dtype=c_norm.dtype)
                    mask = counts_long > 0
                    epoch_counts.add_(counts_long.view(C, M).to(dtype=epoch_counts.dtype))
                    since.masked_fill_(counts_long.view(C, M) > 0, 0)
                    if mask.any():
                        means = sums[mask] / counts[mask].unsqueeze(1)
                        P_view = P.view(num_protos, -1)
                        beta = float(self.proto_ema_momentum)
                        updated = beta * P_view[mask] + (1.0 - beta) * means
                        P_view[mask] = F.normalize(updated, p=2, dim=-1)

                    # Fixed empty-prototype reset (only for classes present in the batch).
                    classes_in_batch = labels.unique()
                    for y in classes_in_batch.tolist():
                        y = int(y)
                        if y < 0 or y >= C:
                            continue
                        member_idx = (labels == y).nonzero(as_tuple=False).view(-1)
                        if member_idx.numel() == 0:
                            continue
                        empty_mask = since[y] > int(self.proto_empty_reset_steps)
                        if not bool(empty_mask.any()):
                            continue
                        for m in torch.nonzero(empty_mask, as_tuple=False).view(-1).tolist():
                            j = int(torch.randint(member_idx.numel(), (1,), device=c.device).item())
                            i = int(member_idx[j].item())
                            P[y, int(m)].copy_(c_norm[i].detach())
                            since[y, int(m)] = 0

        sparsity_pen = self._sparsity_penalty(epoch, total_epochs)
        lambda_w = self._scheduled_lambda(epoch, total_epochs)

        if self.complementarity_enabled and M > 1:
            # Prototype-space redundancy measure for logging/diagnostics.
            P_norm = F.normalize(P, p=2, dim=-1)
            gram = torch.matmul(P_norm, P_norm.transpose(-1, -2))  # [C, M, M]
            eye = torch.eye(M, device=c.device, dtype=gram.dtype).unsqueeze(0)
            off_diag = gram - eye
            complementarity_pen = torch.mean(off_diag ** 2)

            # Lightweight repulsion step on PrototypeBank (no gradients), applied only during training.
            if self.training and self.complementarity_weight > 0.0:
                with torch.no_grad():
                    P_batch = P_norm[labels.unique()]
                    gram_b = torch.matmul(P_batch, P_batch.transpose(-1, -2))
                    eye_b = torch.eye(M, device=c.device, dtype=gram_b.dtype).unsqueeze(0)
                    off_b = gram_b - eye_b
                    repulsion = torch.matmul(off_b, P_batch)
                    P_new = F.normalize(P_batch - float(self.complementarity_weight) * repulsion, p=2, dim=-1)
                    P[labels.unique()] = P_new

        total = L_info + sparsity_pen
        return {
            "total": total,
            "proto_nce": L_info,
            "sparsity_penalty": sparsity_pen,
            "complementarity_penalty": complementarity_pen,
            "lambda_schedule": float(lambda_w),
        }

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
