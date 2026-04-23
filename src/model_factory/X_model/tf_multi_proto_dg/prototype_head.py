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
        assignment_mode: str = "hard",
        balance_weight: float = 0.0,
        logit_scale_init: float = 12.0,
        init_mode: str = "random_normal",
        init_scale: float = 0.02,
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
        self.assignment_mode = str(assignment_mode)
        if self.assignment_mode not in {"hard", "soft_similarity", "soft_balanced"}:
            raise ValueError(
                f"Unsupported assignment_mode={self.assignment_mode!r}. "
                "Expected one of {'hard', 'soft_similarity', 'soft_balanced'}."
            )
        self.balance_weight = float(balance_weight)
        self.logit_scale_init = float(logit_scale_init)
        self.init_mode = str(init_mode)
        if self.init_mode not in {"random_normal", "deterministic_anchor"}:
            raise ValueError(
                f"Unsupported init_mode={self.init_mode!r}. "
                "Expected one of {'random_normal', 'deterministic_anchor'}."
            )
        self.init_scale = float(init_scale)
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

    @staticmethod
    def _build_deterministic_anchor_prototypes(
        class_count: int,
        num_prototypes_per_class: int,
        concept_dim: int,
        scale: float,
    ) -> torch.Tensor:
        dim_index = torch.arange(float(concept_dim), dtype=torch.float32).add(1.0)
        prototypes = torch.zeros(
            int(class_count),
            int(num_prototypes_per_class),
            int(concept_dim),
            dtype=torch.float32,
        )
        target_norm = float(scale) * float(max(concept_dim, 1)) ** 0.5
        for class_id in range(int(class_count)):
            class_scale = float(class_id + 1)
            anchor = (
                torch.sin(dim_index * (0.017 * class_scale) + 0.19 * class_scale)
                + 0.5 * torch.cos(dim_index * (0.011 * class_scale) - 0.23 * class_scale)
            )
            anchor = anchor - anchor.mean()
            anchor = anchor / anchor.norm().clamp_min(1e-6)
            for proto_id in range(int(num_prototypes_per_class)):
                proto_scale = float(proto_id + 1)
                offset = (
                    torch.sin(dim_index * (0.031 * proto_scale) + 0.13 * class_scale)
                    + 0.5 * torch.cos(dim_index * (0.029 * proto_scale) - 0.07 * class_scale)
                )
                offset = offset - offset.mean()
                offset = offset / offset.norm().clamp_min(1e-6)
                vector = anchor + 0.22 * offset
                vector = vector - vector.mean()
                vector = vector / vector.norm().clamp_min(1e-6)
                prototypes[class_id, proto_id] = target_norm * vector
        return prototypes

    def _init_prototypes(self, class_count: int) -> torch.Tensor:
        if self.init_mode == "deterministic_anchor":
            return self._build_deterministic_anchor_prototypes(
                class_count=class_count,
                num_prototypes_per_class=self.num_prototypes_per_class,
                concept_dim=self.concept_dim,
                scale=self.init_scale,
            )
        return torch.randn(class_count, self.num_prototypes_per_class, self.concept_dim) * self.init_scale

    def _register_head(self, head_key: str, class_count: int) -> None:
        bufkey = self._sanitize_key(head_key)
        self._head_to_bufkey[head_key] = bufkey
        proto = nn.Parameter(self._init_prototypes(class_count))
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
        assignment_enabled: bool = True,
        update_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        prototypes, logit_scale_raw, class_bias, _, _, _ = self._get_buffers(head_key)
        tau = max(float(self.temperature), 1e-6)
        assign_tau = max(float(self.assignment_temperature), 1e-6)
        h_norm = F.normalize(h, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        class_anchor = F.normalize(proto_norm.mean(dim=1), dim=-1)
        anchor_scores_raw = torch.einsum("bd,nd->bn", h_norm, class_anchor)
        proto_scores_raw = torch.einsum("bd,nkd->bnk", h_norm, proto_norm)
        proto_residual_scores_raw = proto_scores_raw - anchor_scores_raw.unsqueeze(-1)
        if self.num_prototypes_per_class > 1:
            proto_routing_mean = proto_residual_scores_raw.mean(dim=-1, keepdim=True)
            proto_routing_centered = proto_residual_scores_raw - proto_routing_mean
            proto_routing_scale = proto_routing_centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            proto_routing_scores = proto_routing_centered / proto_routing_scale
        else:
            proto_routing_scores = torch.zeros_like(proto_residual_scores_raw)
        logit_scale = F.softplus(logit_scale_raw) + 1e-4
        anchor_scores = logit_scale * anchor_scores_raw
        proto_scores = logit_scale * proto_residual_scores_raw
        class_effective_k_bias, class_effective_k = self._class_effective_k_bias(
            head_key,
            h.device,
            proto_scores.dtype,
        )
        proto_scores_logits = proto_scores + class_effective_k_bias.unsqueeze(0)
        class_neff = self._class_neff(head_key, h.device, proto_scores.dtype)
        class_temperatures = self._class_temperatures(head_key, h.device, proto_scores.dtype)
        if self.class_pool_mode == "max":
            pooled_residual = proto_scores_logits.max(dim=-1).values
        else:
            pooled_residual = class_temperatures.view(1, -1) * torch.logsumexp(
                proto_scores_logits / class_temperatures.view(1, -1, 1),
                dim=-1,
            )
            pooled_residual = pooled_residual - class_temperatures.view(1, -1) * torch.log(
                torch.tensor(
                    float(max(self.num_prototypes_per_class, 1)),
                    device=proto_scores.device,
                    dtype=proto_scores.dtype,
                )
            )
        logits = anchor_scores + pooled_residual + class_bias.view(1, -1)

        if labels is None:
            target_class_ids = logits.argmax(dim=1)
        else:
            target_class_ids = labels.to(h.device).long()

        gather_index = target_class_ids.view(-1, 1, 1).expand(-1, 1, self.num_prototypes_per_class)
        target_proto_scores = proto_scores.gather(1, gather_index).squeeze(1)
        target_proto_scores_raw = proto_residual_scores_raw.gather(1, gather_index).squeeze(1)
        target_proto_routing_scores = proto_routing_scores.gather(1, gather_index).squeeze(1)
        target_anchor_scores = anchor_scores.gather(1, target_class_ids.view(-1, 1)).squeeze(1)
        if assignment_enabled:
            if self.assignment_mode == "soft_similarity":
                target_proto_scores_routed = target_proto_scores_raw
            else:
                target_proto_scores_routed = target_proto_routing_scores

            if self.assignment_mode == "soft_balanced" and self.balance_weight > 0.0:
                _, _, _, counts, _, _ = self._get_buffers(head_key)
                class_counts = counts.detach().to(
                    device=target_proto_scores_routed.device,
                    dtype=target_proto_scores_routed.dtype,
                )
                target_class_counts = class_counts.index_select(0, target_class_ids)
                target_class_usage = target_class_counts / target_class_counts.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                usage_penalty = -target_class_usage.clamp_min(1e-6).log()
                usage_penalty = usage_penalty - usage_penalty.mean(dim=-1, keepdim=True)
                target_proto_scores_routed = target_proto_scores_routed + self.balance_weight * usage_penalty

            if self.assignment_mode == "hard":
                prototype_assignments = target_proto_scores_routed.argmax(dim=-1)
                target_proto_probs = F.one_hot(
                    prototype_assignments,
                    num_classes=self.num_prototypes_per_class,
                ).to(target_proto_scores_routed.dtype)
            else:
                target_proto_probs = torch.softmax(target_proto_scores_routed / assign_tau, dim=-1)
                prototype_assignments = target_proto_probs.argmax(dim=-1)
            winning_proto_scores = target_proto_scores_raw.gather(
                1, prototype_assignments.view(-1, 1)
            ).squeeze(1)
            if update_enabled:
                self._update_usage(head_key, target_class_ids, target_proto_probs)
        else:
            target_proto_scores_routed = None
            target_proto_probs = None
            prototype_assignments = None
            winning_proto_scores = None
        return {
            "logits": logits,
            "anchor_scores": anchor_scores,
            "anchor_scores_raw": anchor_scores_raw,
            "class_anchor": class_anchor,
            "proto_scores": proto_scores,
            "proto_scores_logits": proto_scores_logits,
            "proto_scores_raw": proto_scores_raw,
            "proto_residual_scores_raw": proto_residual_scores_raw,
            "proto_routing_scores": proto_routing_scores,
            "pooled_residual_scores": pooled_residual,
            "target_class_ids": target_class_ids,
            "class_neff": class_neff,
            "target_class_neff": class_neff.gather(0, target_class_ids),
            "class_temperatures": class_temperatures,
            "target_class_temperatures": class_temperatures.gather(0, target_class_ids),
            "class_effective_k_bias": class_effective_k_bias,
            "class_effective_k": class_effective_k,
            "target_class_effective_k": class_effective_k.gather(0, target_class_ids),
            "target_proto_scores": target_proto_scores,
            "target_proto_scores_raw": target_proto_scores_raw,
            "target_proto_routing_scores": target_proto_routing_scores,
            "target_proto_scores_routed": target_proto_scores_routed,
            "target_anchor_scores": target_anchor_scores,
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
            "assignment_mode": self.assignment_mode,
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
