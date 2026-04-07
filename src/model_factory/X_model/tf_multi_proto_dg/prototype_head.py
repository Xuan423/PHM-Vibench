from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    def __init__(
        self,
        num_classes: int | Dict[str, int],
        concept_dim: int,
        num_prototypes_per_class: int,
        temperature: float = 0.07,
        class_pool_mode: str = "logsumexp",
        adaptive_class_temperature_enabled: bool = False,
        adaptive_class_temperature_target_neff: float = 2.0,
        adaptive_class_temperature_min_scale: float = 0.5,
        adaptive_class_temperature_max_scale: float = 1.5,
        adaptive_effective_k_enabled: bool = False,
        adaptive_effective_k_ready_count: float = 8.0,
        adaptive_effective_k_min: int = 1,
        adaptive_effective_k_penalty: float = 8.0,
        assignment_temperature: float = 0.07,
        logit_scale_init: float = 12.0,
    ) -> None:
        super().__init__()
        self.concept_dim = int(concept_dim)
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.temperature = float(temperature)
        self.class_pool_mode = str(class_pool_mode)
        self.adaptive_class_temperature_enabled = bool(adaptive_class_temperature_enabled)
        self.adaptive_class_temperature_target_neff = float(adaptive_class_temperature_target_neff)
        self.adaptive_class_temperature_min_scale = float(adaptive_class_temperature_min_scale)
        self.adaptive_class_temperature_max_scale = float(adaptive_class_temperature_max_scale)
        self.adaptive_effective_k_enabled = bool(adaptive_effective_k_enabled)
        self.adaptive_effective_k_ready_count = float(adaptive_effective_k_ready_count)
        self.adaptive_effective_k_min = int(adaptive_effective_k_min)
        self.adaptive_effective_k_penalty = float(adaptive_effective_k_penalty)
        self.assignment_temperature = float(assignment_temperature)
        self.logit_scale_init = float(logit_scale_init)
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
        proto = nn.Parameter(torch.randn(class_count, self.num_prototypes_per_class, self.concept_dim) * 0.02)
        self.register_parameter(f"_proto_P_{bufkey}", proto)
        self.register_parameter(
            f"_proto_logit_scale_{bufkey}",
            nn.Parameter(torch.tensor(float(self.logit_scale_init))),
        )
        self.register_parameter(
            f"_proto_class_bias_{bufkey}",
            nn.Parameter(torch.zeros(class_count)),
        )
        self.register_buffer(f"_proto_counts_{bufkey}", torch.zeros(class_count, self.num_prototypes_per_class))
        self.register_buffer(f"_proto_since_{bufkey}", torch.zeros(class_count, self.num_prototypes_per_class))
        self.register_buffer(
            f"_proto_initialized_{bufkey}",
            torch.ones(class_count, self.num_prototypes_per_class, dtype=torch.bool),
        )

    def _get_buffers(
        self, head_key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bufkey = self._head_to_bufkey[str(head_key)]
        return (
            getattr(self, f"_proto_P_{bufkey}"),
            getattr(self, f"_proto_logit_scale_{bufkey}"),
            getattr(self, f"_proto_class_bias_{bufkey}"),
            getattr(self, f"_proto_counts_{bufkey}"),
            getattr(self, f"_proto_since_{bufkey}"),
            getattr(self, f"_proto_initialized_{bufkey}"),
        )

    def _update_usage(self, head_key: str, class_ids: torch.Tensor, target_proto_probs: torch.Tensor) -> None:
        _, _, _, counts, since, initialized = self._get_buffers(head_key)
        counts.mul_(0.99)
        since.add_(1)
        for class_id in class_ids.unique(sorted=True).tolist():
            class_id = int(class_id)
            class_mask = class_ids == class_id
            if class_mask.sum() == 0:
                continue
            counts[class_id] += target_proto_probs[class_mask].detach().sum(dim=0).to(counts.device)
            since[class_id] = 0
            initialized[class_id] = True

    def _class_temperatures(self, head_key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base_tau = max(float(self.temperature), 1e-6)
        prototypes, _, _, _, _, _ = self._get_buffers(head_key)
        num_classes = int(prototypes.shape[0])
        if not self.adaptive_class_temperature_enabled:
            return torch.full((num_classes,), base_tau, device=device, dtype=dtype)

        n_eff = self._class_neff(head_key, device, dtype)
        scale = n_eff / max(self.adaptive_class_temperature_target_neff, 1e-6)
        scale = scale.clamp(
            min=self.adaptive_class_temperature_min_scale,
            max=self.adaptive_class_temperature_max_scale,
        )
        return base_tau * scale

    def _class_neff(self, head_key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        _, _, _, counts, _, _ = self._get_buffers(head_key)
        smoothed = counts.detach().to(device=device, dtype=dtype) + 1.0
        probs = smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
        return torch.exp(entropy)

    def _class_effective_k_bias(
        self,
        head_key: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prototypes, _, _, counts, _, _ = self._get_buffers(head_key)
        num_classes = int(prototypes.shape[0])
        default_k = torch.full(
            (num_classes,),
            self.num_prototypes_per_class,
            device=device,
            dtype=torch.long,
        )
        zero_bias = torch.zeros(
            num_classes,
            self.num_prototypes_per_class,
            device=device,
            dtype=dtype,
        )
        if not self.adaptive_effective_k_enabled or self.num_prototypes_per_class <= 1:
            return zero_bias, default_k

        class_counts = counts.detach().to(device=device, dtype=dtype)
        n_eff = self._class_neff(head_key, device, dtype)
        effective_k = torch.round(n_eff).to(dtype=torch.long)
        effective_k = effective_k.clamp(
            min=max(int(self.adaptive_effective_k_min), 1),
            max=self.num_prototypes_per_class,
        )
        ready = (
            class_counts.sum(dim=-1)
            / (
                class_counts.sum(dim=-1)
                + max(float(self.adaptive_effective_k_ready_count), 1e-6)
            )
        ).clamp(min=0.0, max=1.0)
        order = torch.argsort(class_counts, dim=-1, descending=True)
        ranks = torch.argsort(order, dim=-1)
        inactive = ranks >= effective_k.view(-1, 1)
        penalty = float(self.adaptive_effective_k_penalty) * ready.view(-1, 1)
        return -penalty * inactive.to(dtype), effective_k

    def forward(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None = None,
        head_key: str = "default",
    ) -> Dict[str, torch.Tensor]:
        prototypes, logit_scale_raw, class_bias, _, _, _ = self._get_buffers(head_key)
        tau = max(float(self.temperature), 1e-6)
        assign_tau = max(float(self.assignment_temperature), 1e-6)
        h_norm = F.normalize(h, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        proto_scores_raw = torch.einsum("bd,nkd->bnk", h_norm, proto_norm)
        logit_scale = F.softplus(logit_scale_raw) + 1e-4
        proto_scores = logit_scale * proto_scores_raw
        class_effective_k_bias, class_effective_k = self._class_effective_k_bias(
            head_key,
            h.device,
            proto_scores.dtype,
        )
        proto_scores_logits = proto_scores + class_effective_k_bias.unsqueeze(0)
        class_neff = self._class_neff(head_key, h.device, proto_scores.dtype)
        class_temperatures = self._class_temperatures(head_key, h.device, proto_scores.dtype)
        if self.class_pool_mode == "max":
            pooled_scores = proto_scores_logits.max(dim=-1).values
        else:
            pooled_scores = class_temperatures.view(1, -1) * torch.logsumexp(
                proto_scores_logits / class_temperatures.view(1, -1, 1),
                dim=-1,
            )
        logits = pooled_scores + class_bias.view(1, -1)

        if labels is None:
            target_class_ids = logits.argmax(dim=1)
        else:
            target_class_ids = labels.to(h.device).long()

        gather_index = target_class_ids.view(-1, 1, 1).expand(-1, 1, self.num_prototypes_per_class)
        target_proto_scores = proto_scores.gather(1, gather_index).squeeze(1)
        target_proto_scores_routed = target_proto_scores
        target_proto_probs = torch.softmax(target_proto_scores_routed / assign_tau, dim=-1)
        prototype_assignments = target_proto_probs.argmax(dim=-1)
        winning_proto_scores = target_proto_scores.gather(
            1, prototype_assignments.view(-1, 1)
        ).squeeze(1)
        self._update_usage(head_key, target_class_ids, target_proto_probs)
        return {
            "logits": logits,
            "proto_scores": proto_scores,
            "proto_scores_logits": proto_scores_logits,
            "proto_scores_raw": proto_scores_raw,
            "target_class_ids": target_class_ids,
            "class_neff": class_neff,
            "target_class_neff": class_neff.gather(0, target_class_ids),
            "class_temperatures": class_temperatures,
            "target_class_temperatures": class_temperatures.gather(0, target_class_ids),
            "class_effective_k_bias": class_effective_k_bias,
            "class_effective_k": class_effective_k,
            "target_class_effective_k": class_effective_k.gather(0, target_class_ids),
            "target_proto_scores": target_proto_scores,
            "target_proto_scores_routed": target_proto_scores_routed,
            "target_proto_probs": target_proto_probs,
            "prototype_assignments": prototype_assignments,
            "prototype_assignment_weights": target_proto_probs,
            "prototype_pos_scores": winning_proto_scores,
        }

    def diversity_penalty(self, head_key: str) -> torch.Tensor:
        prototypes, _, _, _, _, _ = self._get_buffers(head_key)
        normalized = F.normalize(prototypes, dim=-1)
        pairwise = torch.matmul(normalized, normalized.transpose(-1, -2))
        if pairwise.shape[-1] <= 1:
            return prototypes.new_zeros(())
        eye = torch.eye(pairwise.shape[-1], device=pairwise.device, dtype=torch.bool).unsqueeze(0)
        offdiag = pairwise.masked_select(~eye)
        if offdiag.numel() == 0:
            return prototypes.new_zeros(())
        return offdiag.clamp_min(0.0).pow(2).mean()

    def diversity_excess(self, head_key: str, target_cos: float = 0.0) -> torch.Tensor:
        prototypes, _, _, _, _, _ = self._get_buffers(head_key)
        normalized = F.normalize(prototypes, dim=-1)
        pairwise = torch.matmul(normalized, normalized.transpose(-1, -2))
        if pairwise.shape[-1] <= 1:
            return prototypes.new_zeros(())
        eye = torch.eye(pairwise.shape[-1], device=pairwise.device, dtype=torch.bool).unsqueeze(0)
        offdiag = pairwise.masked_select(~eye)
        if offdiag.numel() == 0:
            return prototypes.new_zeros(())
        return offdiag.sub(float(target_cos)).clamp_min(0.0).mean()

    def export_health(self, head_key: str) -> Dict[str, list | str | int]:
        prototypes, logit_scale_raw, _, counts, _, initialized = self._get_buffers(head_key)
        sum_counts = counts.sum(dim=1)
        assignment_ratio = torch.where(
            sum_counts > 0,
            counts.max(dim=1).values / sum_counts.clamp_min(1e-6),
            torch.zeros_like(sum_counts),
        )
        probs = counts / sum_counts.unsqueeze(1).clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=1)
        n_eff = torch.where(sum_counts > 0, torch.exp(entropy), torch.zeros_like(sum_counts))
        normalized = F.normalize(prototypes.detach(), dim=-1)
        pairwise = torch.matmul(normalized, normalized.transpose(-1, -2))
        if pairwise.shape[1] > 1:
            eye = torch.eye(pairwise.shape[-1], device=pairwise.device, dtype=torch.bool).unsqueeze(0)
            max_offdiag = pairwise.masked_fill(eye, float("-inf")).max(dim=-1).values.max(dim=-1).values
            max_offdiag = torch.where(torch.isfinite(max_offdiag), max_offdiag, torch.zeros_like(max_offdiag))
        else:
            max_offdiag = torch.zeros(prototypes.shape[0], device=prototypes.device)
        class_temperatures = self._class_temperatures(
            head_key,
            prototypes.device,
            prototypes.dtype,
        )
        _, class_effective_k = self._class_effective_k_bias(
            head_key,
            prototypes.device,
            prototypes.dtype,
        )
        return {
            "assignment_mode": "lsep_simplified",
            "num_classes": int(prototypes.shape[0]),
            "M": int(prototypes.shape[1]),
            "initialized": initialized.detach().cpu().int().tolist(),
            "max_assignment_ratio": assignment_ratio.detach().cpu().tolist(),
            "n_eff": n_eff.detach().cpu().tolist(),
            "class_temperatures": class_temperatures.detach().cpu().tolist(),
            "class_effective_k": class_effective_k.detach().cpu().tolist(),
            "max_offdiag_cos": max_offdiag.detach().cpu().tolist(),
            "logit_scale": float((F.softplus(logit_scale_raw) + 1e-4).detach().cpu().item()),
            "pairwise_cos": pairwise.detach().cpu().tolist(),
        }

    def export_all_health(self) -> Dict[str, Dict[str, list | str | int]]:
        return {head_key: self.export_health(head_key) for head_key in self.head_keys}
