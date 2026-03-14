from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_classes: int | Dict[str, int],
        concept_dim: int,
        num_prototypes_per_class: int,
        ema_momentum: float = 0.95,
        neg_k: int = 0,
        empty_reset_steps: int = 200,
    ) -> None:
        super().__init__()
        self.concept_dim = int(concept_dim)
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.ema_momentum = float(ema_momentum)
        self.neg_k = int(neg_k)
        self.empty_reset_steps = int(empty_reset_steps)
        self.head_to_num_classes = (
            {str(key): int(value) for key, value in num_classes.items()}
            if isinstance(num_classes, dict)
            else {"default": int(num_classes)}
        )
        self._head_to_bufkey: Dict[str, str] = {}
        for head_key, class_count in self.head_to_num_classes.items():
            self._register_head(head_key, class_count)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in str(key))

    @property
    def head_keys(self) -> List[str]:
        return list(self.head_to_num_classes.keys())

    def _register_head(self, head_key: str, class_count: int) -> None:
        bufkey = self._sanitize_key(head_key)
        self._head_to_bufkey[head_key] = bufkey
        self.register_buffer(
            f"_proto_P_{bufkey}",
            F.normalize(torch.randn(class_count, self.num_prototypes_per_class, self.concept_dim), dim=-1),
        )
        self.register_buffer(f"_proto_counts_{bufkey}", torch.zeros(class_count, self.num_prototypes_per_class))
        self.register_buffer(f"_proto_since_{bufkey}", torch.zeros(class_count, self.num_prototypes_per_class))
        self.register_buffer(f"_proto_initialized_{bufkey}", torch.zeros(class_count, self.num_prototypes_per_class, dtype=torch.bool))

    def _get_buffers(self, head_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bufkey = self._head_to_bufkey[str(head_key)]
        return (
            getattr(self, f"_proto_P_{bufkey}"),
            getattr(self, f"_proto_counts_{bufkey}"),
            getattr(self, f"_proto_since_{bufkey}"),
            getattr(self, f"_proto_initialized_{bufkey}"),
        )

    def _snapshot_head(self, head_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        P, _, _, initialized = self._get_buffers(head_key)
        return P.detach().clone(), initialized.detach().clone()

    def assign(self, c: torch.Tensor, labels: torch.Tensor, head_key: str) -> Dict[str, torch.Tensor]:
        P, counts, since, initialized = self._get_buffers(head_key)
        since.add_(1)
        assignments = torch.zeros(labels.shape[0], device=c.device, dtype=torch.long)
        pos_scores = torch.zeros(labels.shape[0], device=c.device, dtype=c.dtype)

        for class_id in labels.unique(sorted=True).tolist():
            class_id = int(class_id)
            class_mask = labels == class_id
            class_c = c[class_mask]
            if class_c.numel() == 0:
                continue
            initialized_row = initialized[class_id]
            uninit = (~initialized_row).nonzero(as_tuple=False).view(-1)
            if uninit.numel() > 0:
                take = min(uninit.numel(), class_c.shape[0])
                P[class_id, uninit[:take]] = class_c[:take].detach()
                initialized[class_id, uninit[:take]] = True

            valid_idx = initialized[class_id].nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                valid_idx = torch.tensor([0], device=c.device, dtype=torch.long)
                P[class_id, 0] = class_c[0].detach()
                initialized[class_id, 0] = True

            sims = torch.matmul(class_c, P[class_id, valid_idx].transpose(0, 1))
            local_assign = sims.argmax(dim=1)
            global_proto = valid_idx[local_assign]
            assignments[class_mask] = global_proto
            pos_scores[class_mask] = sims.gather(1, local_assign.unsqueeze(1)).squeeze(1)
        return {"assignments": assignments, "pos_scores": pos_scores}

    def _flatten_active_prototypes(self, head_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        P, _, _, initialized = self._get_buffers(head_key)
        return self._flatten_active_prototypes_from_snapshot(P, initialized)

    def _flatten_active_prototypes_from_snapshot(
        self, P: torch.Tensor, initialized: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        protos = []
        proto_classes = []
        proto_slots = []
        for class_id in range(P.shape[0]):
            valid_idx = initialized[class_id].nonzero(as_tuple=False).view(-1)
            for proto_id in valid_idx.tolist():
                protos.append(P[class_id, proto_id])
                proto_classes.append(class_id)
                proto_slots.append(proto_id)
        if not protos:
            return (
                torch.empty(0, self.concept_dim, device=P.device),
                torch.empty(0, dtype=torch.long, device=P.device),
                torch.empty(0, dtype=torch.long, device=P.device),
            )
        return (
            torch.stack(protos, dim=0),
            torch.as_tensor(proto_classes, device=P.device, dtype=torch.long),
            torch.as_tensor(proto_slots, device=P.device, dtype=torch.long),
        )

    def compute_losses(
        self,
        c: torch.Tensor,
        labels: torch.Tensor,
        assignments: torch.Tensor,
        head_key: str,
        metric_weight: torch.Tensor,
        temperature: float,
        readability_weight: float,
        complementarity_weight: float,
    ) -> Dict[str, torch.Tensor]:
        prototype_snapshot, initialized_snapshot = self._snapshot_head(head_key)
        prototypes, proto_classes, proto_slots = self._flatten_active_prototypes_from_snapshot(
            prototype_snapshot, initialized_snapshot
        )
        if prototypes.numel() == 0:
            zero = c.new_zeros(())
            return {
                "total": zero,
                "proto_nce": zero,
                "readability_penalty": zero,
                "complementarity_penalty": zero,
                "assignment_ratio": zero,
            }

        sims = torch.matmul(c, prototypes.transpose(0, 1))
        pos_indices = []
        for label, slot in zip(labels.tolist(), assignments.tolist()):
            match = ((proto_classes == int(label)) & (proto_slots == int(slot))).nonzero(as_tuple=False).view(-1)
            pos_indices.append(int(match[0].item()) if match.numel() > 0 else 0)
        pos_indices_tensor = torch.as_tensor(pos_indices, device=c.device, dtype=torch.long)

        losses = []
        for sample_idx in range(c.shape[0]):
            sample_sims = sims[sample_idx]
            pos_score = sample_sims[pos_indices_tensor[sample_idx]].view(1)
            neg_scores = sample_sims[proto_classes != labels[sample_idx]]
            if self.neg_k > 0 and neg_scores.numel() > self.neg_k:
                neg_scores = torch.topk(neg_scores, k=self.neg_k).values
            logits = torch.cat([pos_score, neg_scores], dim=0) / max(float(temperature), 1e-6)
            losses.append(-F.log_softmax(logits, dim=0)[0])
        proto_nce = torch.stack(losses).mean() if losses else c.new_zeros(())

        signatures = self.compute_signatures_from_snapshot(
            prototype_snapshot,
            initialized_snapshot,
            metric_weight,
        )
        readability_penalty = self.compute_readability_penalty(signatures) * float(readability_weight)
        complementarity_penalty = self.compute_complementarity_penalty(signatures) * float(
            complementarity_weight
        )

        bincount = torch.bincount(assignments, minlength=self.num_prototypes_per_class).float()
        assignment_ratio = bincount.max() / bincount.sum().clamp_min(1.0)
        total = proto_nce + readability_penalty + complementarity_penalty
        return {
            "total": total,
            "proto_nce": proto_nce,
            "readability_penalty": readability_penalty,
            "complementarity_penalty": complementarity_penalty,
            "assignment_ratio": assignment_ratio,
        }

    def update(self, c: torch.Tensor, labels: torch.Tensor, assignments: torch.Tensor, head_key: str) -> None:
        P, counts, since, initialized = self._get_buffers(head_key)
        for class_id in labels.unique(sorted=True).tolist():
            class_id = int(class_id)
            class_mask = labels == class_id
            class_assignments = assignments[class_mask]
            class_c = c[class_mask]
            for proto_id in class_assignments.unique(sorted=True).tolist():
                proto_id = int(proto_id)
                proto_mask = class_assignments == proto_id
                if proto_mask.sum() == 0:
                    continue
                proto_mean = F.normalize(class_c[proto_mask].mean(dim=0), dim=0)
                if initialized[class_id, proto_id]:
                    P[class_id, proto_id] = F.normalize(
                        self.ema_momentum * P[class_id, proto_id] + (1.0 - self.ema_momentum) * proto_mean,
                        dim=0,
                    )
                else:
                    P[class_id, proto_id] = proto_mean
                    initialized[class_id, proto_id] = True
                counts[class_id, proto_id] += float(proto_mask.sum().item())
                since[class_id, proto_id] = 0

        for class_id in labels.unique(sorted=True).tolist():
            class_id = int(class_id)
            class_mask = labels == class_id
            if class_mask.sum() == 0:
                continue
            class_c = c[class_mask]
            valid_idx = initialized[class_id].nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue
            stale_idx = (since[class_id, valid_idx] >= self.empty_reset_steps).nonzero(as_tuple=False).view(-1)
            if stale_idx.numel() == 0:
                continue
            sims = torch.matmul(class_c, P[class_id, valid_idx].transpose(0, 1))
            hardest = sims.max(dim=1).values.argmin()
            for stale_local in stale_idx.tolist():
                proto_id = int(valid_idx[stale_local].item())
                P[class_id, proto_id] = class_c[hardest].detach()
                since[class_id, proto_id] = 0
                initialized[class_id, proto_id] = True

    def compute_signatures(self, head_key: str, metric_weight: torch.Tensor) -> torch.Tensor:
        P, _, _, initialized = self._get_buffers(head_key)
        return self.compute_signatures_from_snapshot(P, initialized, metric_weight)

    @staticmethod
    def compute_signatures_from_snapshot(
        P: torch.Tensor, initialized: torch.Tensor, metric_weight: torch.Tensor
    ) -> torch.Tensor:
        signatures = torch.abs(torch.matmul(P, metric_weight))
        signatures = torch.where(initialized.unsqueeze(-1), signatures, torch.zeros_like(signatures))
        return signatures

    @staticmethod
    def compute_readability_penalty(signatures: torch.Tensor) -> torch.Tensor:
        probs = signatures / signatures.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
        max_entropy = torch.log(torch.as_tensor(signatures.shape[-1], device=signatures.device, dtype=signatures.dtype))
        return (entropy / max_entropy.clamp_min(1e-6)).mean()

    @staticmethod
    def compute_complementarity_penalty(signatures: torch.Tensor) -> torch.Tensor:
        if signatures.shape[1] <= 1:
            return signatures.new_zeros(())
        sig = F.normalize(signatures, dim=-1)
        pairwise = torch.matmul(sig, sig.transpose(-1, -2))
        eye = torch.eye(pairwise.shape[-1], device=pairwise.device, dtype=torch.bool).unsqueeze(0)
        offdiag = pairwise.masked_fill(eye, 0.0)
        return offdiag.square().mean()

    def export_health(self, head_key: str) -> Dict[str, list]:
        P, counts, _, initialized = self._get_buffers(head_key)
        sum_counts = counts.sum(dim=1)
        max_counts = counts.max(dim=1).values
        assignment_ratio = torch.where(sum_counts > 0, max_counts / sum_counts.clamp_min(1e-6), torch.zeros_like(sum_counts))
        probs = counts / sum_counts.unsqueeze(1).clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=1)
        n_eff = torch.where(sum_counts > 0, torch.exp(entropy), torch.zeros_like(sum_counts))
        signatures = F.normalize(P, dim=-1)
        if signatures.shape[1] > 1:
            pairwise = torch.matmul(signatures, signatures.transpose(-1, -2))
            eye = torch.eye(pairwise.shape[-1], device=pairwise.device, dtype=torch.bool).unsqueeze(0)
            max_offdiag = pairwise.masked_fill(eye, float("-inf")).max(dim=-1).values.max(dim=-1).values
            max_offdiag = torch.where(torch.isfinite(max_offdiag), max_offdiag, torch.zeros_like(max_offdiag))
        else:
            pairwise = torch.ones(P.shape[0], 1, 1, device=P.device)
            max_offdiag = torch.zeros(P.shape[0], device=P.device)
        return {
            "num_classes": int(P.shape[0]),
            "M": int(P.shape[1]),
            "initialized": initialized.detach().cpu().int().tolist(),
            "max_assignment_ratio": assignment_ratio.detach().cpu().tolist(),
            "n_eff": n_eff.detach().cpu().tolist(),
            "max_offdiag_cos": max_offdiag.detach().cpu().tolist(),
            "pairwise_cos": pairwise.detach().cpu().tolist(),
        }

    def export_all_health(self) -> Dict[str, Dict[str, list]]:
        return {head_key: self.export_health(head_key) for head_key in self.head_keys}
