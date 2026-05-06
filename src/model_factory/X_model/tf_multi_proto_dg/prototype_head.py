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
        class_anchor_mode: str = "mean",
        class_anchor_memory_weight: float = 0.0,
        class_anchor_memory_momentum: float = 0.95,
        anchor_score_mode: str = "full",
        residual_score_mode: str = "dot_difference",
        residual_logit_mode: str = "static",
        residual_logit_weight: float = 1.0,
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
        self.class_anchor_mode = str(class_anchor_mode)
        self.class_anchor_memory_weight = float(class_anchor_memory_weight)
        self.class_anchor_memory_momentum = float(class_anchor_memory_momentum)
        if self.class_anchor_mode not in {
            "mean",
            "independent",
            "tied_offset",
            "hybrid_tied_mean",
        }:
            raise ValueError(
                f"Unsupported class_anchor_mode={self.class_anchor_mode!r}. "
                "Expected one of {'mean', 'independent', 'tied_offset', 'hybrid_tied_mean'}."
            )
        self.anchor_score_mode = str(anchor_score_mode)
        if self.anchor_score_mode not in {
            "full",
            "time_tf",
        }:
            raise ValueError(
                f"Unsupported anchor_score_mode={self.anchor_score_mode!r}. "
                "Expected one of {'full', 'time_tf'}."
            )
        self.residual_score_mode = str(residual_score_mode)
        if self.residual_score_mode not in {"dot_difference", "anchored_offset", "anchored_routing"}:
            raise ValueError(
                f"Unsupported residual_score_mode={self.residual_score_mode!r}. "
                "Expected one of {'dot_difference', 'anchored_offset', 'anchored_routing'}."
            )
        self.residual_logit_mode = str(residual_logit_mode)
        if self.residual_logit_mode not in {
            "static",
            "agreement",
            "agreement_tf_balance",
            "local_evidence",
            "agreement_local_centered",
            "agreement_local_slot_verify",
            "local_competition",
            "global_local_consensus",
            "agreement_global_local_consensus",
            "agreement_anchor_support",
            "anchor_prior",
            "agreement_anchor_prior",
            "anchor_uncertainty_centered",
            "agreement_anchor_uncertainty_centered",
            "agreement_candidate_centered",
            "anchor_residual_margin_mix",
            "agreement_anchor_residual_margin_mix",
            "agreement_candidate_margin_mix",
        }:
            raise ValueError(
                f"Unsupported residual_logit_mode={self.residual_logit_mode!r}."
            )
        self.residual_logit_weight = float(residual_logit_weight)
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
        initial_proto = self._init_prototypes(class_count)
        anchor_init = initial_proto.mean(dim=1)
        if self.class_anchor_mode in {"tied_offset", "hybrid_tied_mean"}:
            proto = nn.Parameter(initial_proto - anchor_init.unsqueeze(1))
        else:
            proto = nn.Parameter(initial_proto)
        self.register_parameter(f"_proto_P_{bufkey}", proto)
        if self.class_anchor_mode in {"independent", "tied_offset", "hybrid_tied_mean"}:
            self.register_parameter(
                f"_proto_class_anchor_{bufkey}",
                nn.Parameter(anchor_init.detach().clone()),
            )
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
        self.register_buffer(
            f"_proto_class_anchor_memory_{bufkey}",
            F.normalize(anchor_init.detach().clone(), dim=-1),
        )
        self.register_buffer(
            f"_proto_class_anchor_memory_initialized_{bufkey}",
            torch.zeros(class_count, dtype=torch.bool),
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

    def _get_anchor_memory(self, head_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        bufkey = self._head_to_bufkey[str(head_key)]
        return (
            getattr(self, f"_proto_class_anchor_memory_{bufkey}"),
            getattr(self, f"_proto_class_anchor_memory_initialized_{bufkey}"),
        )

    @torch.no_grad()
    def _update_anchor_memory(
        self,
        head_key: str,
        labels: torch.Tensor,
        anchor_h_norm: torch.Tensor,
    ) -> None:
        if self.class_anchor_memory_weight <= 0.0:
            return
        memory, initialized = self._get_anchor_memory(head_key)
        momentum = min(max(float(self.class_anchor_memory_momentum), 0.0), 0.9999)
        labels = labels.to(anchor_h_norm.device).long()
        for class_id in labels.unique(sorted=True).tolist():
            class_id = int(class_id)
            class_mask = labels == class_id
            if class_mask.sum() == 0 or class_id < 0 or class_id >= memory.shape[0]:
                continue
            class_mean = F.normalize(anchor_h_norm[class_mask].mean(dim=0), dim=0)
            if bool(initialized[class_id].item()):
                updated = F.normalize(
                    momentum * memory[class_id].to(anchor_h_norm.device)
                    + (1.0 - momentum) * class_mean,
                    dim=0,
                )
            else:
                updated = class_mean
            memory[class_id].copy_(updated.to(memory.device, dtype=memory.dtype))
            initialized[class_id] = True

    def _blend_anchor_memory(
        self,
        head_key: str,
        class_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if self.class_anchor_memory_weight <= 0.0 or self.training:
            return class_anchor
        memory, initialized = self._get_anchor_memory(head_key)
        memory_snapshot = memory.detach().clone()
        initialized_snapshot = initialized.detach().clone()
        weight = class_anchor.new_full(
            (class_anchor.shape[0], 1),
            min(max(float(self.class_anchor_memory_weight), 0.0), 1.0),
        )
        mask = initialized_snapshot.to(device=class_anchor.device).view(-1, 1)
        if not bool(mask.any().item()):
            return class_anchor
        memory_snapshot = memory_snapshot.to(device=class_anchor.device, dtype=class_anchor.dtype)
        blended = F.normalize(
            (1.0 - weight) * class_anchor + weight * memory_snapshot,
            dim=-1,
        )
        return torch.where(mask, blended, class_anchor)

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

    def _materialize_anchor_and_prototypes(
        self,
        head_key: str,
        prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the normalized class-anchor and prototype geometry.

        ``tied_offset`` parameterizes prototypes as zero-mean class-internal
        offsets around an explicit class anchor. ``hybrid_tied_mean`` keeps that
        offset geometry, but uses the consensus direction between the explicit
        anchor and the normalized prototype centroid as the class anchor. This
        preserves a single semantic anchor while allowing the prototype evidence
        to stabilize anchor orientation.
        """
        if self.class_anchor_mode in {"tied_offset", "hybrid_tied_mean"}:
            class_anchor_param = getattr(
                self,
                f"_proto_class_anchor_{self._head_to_bufkey[str(head_key)]}",
            )
            offset = prototypes - prototypes.mean(dim=1, keepdim=True)
            materialized = class_anchor_param.unsqueeze(1) + offset
            proto_norm = F.normalize(materialized, dim=-1)
            class_anchor = F.normalize(class_anchor_param, dim=-1)
            if self.class_anchor_mode == "hybrid_tied_mean":
                proto_centroid = F.normalize(proto_norm.mean(dim=1), dim=-1)
                class_anchor = F.normalize(class_anchor + proto_centroid, dim=-1)
            return class_anchor, proto_norm

        proto_norm = F.normalize(prototypes, dim=-1)
        if self.class_anchor_mode == "independent":
            class_anchor_param = getattr(
                self,
                f"_proto_class_anchor_{self._head_to_bufkey[str(head_key)]}",
            )
            class_anchor = F.normalize(class_anchor_param, dim=-1)
        else:
            class_anchor = F.normalize(proto_norm.mean(dim=1), dim=-1)
        return class_anchor, proto_norm

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

    @staticmethod
    def _class_margin(scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[1] <= 1:
            return torch.zeros(scores.shape[0], device=scores.device, dtype=scores.dtype)
        top2 = scores.topk(k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def _pool_class_residual(
        self,
        proto_scores_logits: torch.Tensor,
        class_temperatures: torch.Tensor,
    ) -> torch.Tensor:
        if self.class_pool_mode == "max":
            return proto_scores_logits.max(dim=-1).values
        if self.class_pool_mode == "mean":
            return proto_scores_logits.mean(dim=-1)
        pooled = class_temperatures.view(1, -1) * torch.logsumexp(
            proto_scores_logits / class_temperatures.view(1, -1, 1),
            dim=-1,
        )
        return pooled - class_temperatures.view(1, -1) * torch.log(
            torch.tensor(
                float(max(self.num_prototypes_per_class, 1)),
                device=proto_scores_logits.device,
                dtype=proto_scores_logits.dtype,
            )
        )

    def _calibrate_residual_for_anchor(
        self,
        pooled_residual: torch.Tensor,
        anchor_scores: torch.Tensor,
        residual_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constrain prototype residuals to explain offsets around the anchor.

        Anchor-prior modes keep residuals on anchor-plausible classes.
        Anchor-uncertainty-centered modes additionally remove the residual
        expectation under the anchor distribution and let the residual act only
        when the anchor itself is uncertain. This keeps the class anchor as the
        single semantic decision path while preserving prototype corrections as
        interpretable class-internal evidence.
        """
        if self.residual_logit_mode in {"anchor_prior", "agreement_anchor_prior"}:
            anchor_prior = torch.softmax(anchor_scores.detach(), dim=1)
            class_count = float(max(int(anchor_scores.shape[1]), 1))
            return pooled_residual, residual_weight * anchor_prior * class_count

        if self.residual_logit_mode in {
            "anchor_uncertainty_centered",
            "agreement_anchor_uncertainty_centered",
        }:
            anchor_prior = torch.softmax(anchor_scores.detach(), dim=1)
            prior_mean = (pooled_residual * anchor_prior).sum(dim=1, keepdim=True)
            pooled_residual = pooled_residual - prior_mean
            class_count = float(max(int(anchor_scores.shape[1]), 1))
            if class_count <= 1.0:
                uncertainty = torch.zeros_like(prior_mean)
            else:
                max_prior = anchor_prior.max(dim=1, keepdim=True).values
                uncertainty = ((1.0 - max_prior) * class_count / (class_count - 1.0)).clamp(
                    min=0.0,
                    max=1.0,
                )
            return pooled_residual, residual_weight * uncertainty

        return pooled_residual, residual_weight

    @staticmethod
    def _anchored_offset_scores(
        concept_norm: torch.Tensor,
        anchor_norm: torch.Tensor,
        proto_norm: torch.Tensor,
        class_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Score prototype deviations around the same class anchor used by logits.

        The class-level decision stays on `anchor_norm -> class_anchor`; prototype
        residuals only explain how the fused/local concept deviates from that
        anchor. This avoids a second fused class-anchor path competing with the
        global anchor.
        """
        concept_delta = concept_norm - anchor_norm
        delta_norm = concept_delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        concept_delta_dir = concept_delta / delta_norm
        proto_offset = F.normalize(
            proto_norm - class_anchor.unsqueeze(1),
            dim=-1,
        )
        return delta_norm.unsqueeze(1) * torch.einsum(
            "bd,nkd->bnk",
            concept_delta_dir,
            proto_offset,
        )

    def _class_anchor_scores(
        self,
        concept_norm: torch.Tensor,
        class_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if int(concept_norm.shape[-1]) != int(class_anchor.shape[-1]):
            return torch.einsum("bd,nd->bn", concept_norm, class_anchor)
        if (
            self.anchor_score_mode == "time_tf"
            and int(concept_norm.shape[-1]) % 4 == 0
        ):
            role_dim = int(concept_norm.shape[-1]) // 4
            concept_segments = concept_norm.view(concept_norm.shape[0], 4, role_dim)
            anchor_segments = class_anchor.view(class_anchor.shape[0], 4, role_dim)
            segment_scores = torch.einsum(
                "bsd,csd->bcs",
                concept_segments,
                anchor_segments,
            )
            return segment_scores[:, :, 0] + segment_scores[:, :, 2]
        return torch.einsum("bd,nd->bn", concept_norm, class_anchor)

    def forward(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None = None,
        head_key: str = "default",
        assignment_enabled: bool = True,
        update_enabled: bool = True,
        anchor_h: torch.Tensor | None = None,
        alternative_anchor_h: torch.Tensor | None = None,
        anchor_mix_mode: str = "adaptive_margin",
        residual_logit_weight: torch.Tensor | None = None,
        routing_h: torch.Tensor | None = None,
        candidate_competition: bool = False,
    ) -> Dict[str, torch.Tensor]:
        prototypes, logit_scale_raw, class_bias, _, _, _ = self._get_buffers(head_key)
        tau = max(float(self.temperature), 1e-6)
        assign_tau = max(float(self.assignment_temperature), 1e-6)
        h_norm = F.normalize(h, dim=-1)
        anchor_h_norm = h_norm if anchor_h is None else F.normalize(anchor_h, dim=-1)
        class_anchor, proto_norm = self._materialize_anchor_and_prototypes(head_key, prototypes)
        class_anchor = self._blend_anchor_memory(head_key, class_anchor)
        full_anchor_scores_raw = torch.einsum("bd,nd->bn", anchor_h_norm, class_anchor)
        anchor_scores_raw = self._class_anchor_scores(anchor_h_norm, class_anchor)
        proto_scores_raw = torch.einsum("bd,nkd->bnk", h_norm, proto_norm)
        anchored_proto_residual_scores_raw = None
        if self.residual_score_mode in {"anchored_offset", "anchored_routing"} and anchor_h is not None:
            anchored_proto_residual_scores_raw = self._anchored_offset_scores(
                h_norm,
                anchor_h_norm,
                proto_norm,
                class_anchor,
            )
        if self.residual_score_mode == "anchored_offset" and anchored_proto_residual_scores_raw is not None:
            proto_residual_scores_raw = anchored_proto_residual_scores_raw
        else:
            proto_residual_scores_raw = proto_scores_raw - full_anchor_scores_raw.unsqueeze(-1)
        routing_proto_residual_scores_raw = (
            anchored_proto_residual_scores_raw
            if self.residual_score_mode in {"anchored_offset", "anchored_routing"}
            and anchored_proto_residual_scores_raw is not None
            else proto_residual_scores_raw
        )
        if routing_h is not None:
            routing_h_norm = F.normalize(routing_h, dim=-1)
            local_proto_scores_raw = torch.einsum("bd,nkd->bnk", routing_h_norm, proto_norm)
            local_anchor_scores_raw = torch.einsum("bd,nd->bn", routing_h_norm, class_anchor)
            anchored_local_proto_residual_scores_raw = None
            if self.residual_score_mode in {"anchored_offset", "anchored_routing"} and anchor_h is not None:
                anchored_local_proto_residual_scores_raw = self._anchored_offset_scores(
                    routing_h_norm,
                    anchor_h_norm,
                    proto_norm,
                    class_anchor,
                )
            if (
                self.residual_score_mode == "anchored_offset"
                and anchored_local_proto_residual_scores_raw is not None
            ):
                local_proto_residual_scores_raw = anchored_local_proto_residual_scores_raw
            else:
                local_proto_residual_scores_raw = (
                    local_proto_scores_raw - local_anchor_scores_raw.unsqueeze(-1)
                )
            routing_local_proto_residual_scores_raw = (
                anchored_local_proto_residual_scores_raw
                if self.residual_score_mode in {"anchored_offset", "anchored_routing"}
                and anchored_local_proto_residual_scores_raw is not None
                else local_proto_residual_scores_raw
            )
        else:
            local_proto_residual_scores_raw = proto_residual_scores_raw
            routing_local_proto_residual_scores_raw = routing_proto_residual_scores_raw
        if self.num_prototypes_per_class > 1:
            proto_routing_mean = routing_proto_residual_scores_raw.mean(dim=-1, keepdim=True)
            proto_routing_centered = routing_proto_residual_scores_raw - proto_routing_mean
            proto_routing_scale = proto_routing_centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            proto_routing_scores = proto_routing_centered / proto_routing_scale
            local_proto_routing_mean = routing_local_proto_residual_scores_raw.mean(dim=-1, keepdim=True)
            local_proto_routing_centered = routing_local_proto_residual_scores_raw - local_proto_routing_mean
            local_proto_routing_scale = (
                local_proto_routing_centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            )
            local_proto_routing_scores = local_proto_routing_centered / local_proto_routing_scale
        else:
            proto_routing_scores = torch.zeros_like(proto_residual_scores_raw)
            local_proto_routing_scores = torch.zeros_like(local_proto_residual_scores_raw)
        logit_scale = F.softplus(logit_scale_raw) + 1e-4
        anchor_scores = logit_scale * anchor_scores_raw
        proto_scores = logit_scale * proto_residual_scores_raw
        local_proto_scores = logit_scale * local_proto_residual_scores_raw
        class_effective_k_bias, class_effective_k = self._class_effective_k_bias(
            head_key,
            h.device,
            proto_scores.dtype,
        )
        proto_scores_logits = proto_scores + class_effective_k_bias.unsqueeze(0)
        local_proto_scores_logits = local_proto_scores + class_effective_k_bias.unsqueeze(0)
        class_neff = self._class_neff(head_key, h.device, proto_scores.dtype)
        class_temperatures = self._class_temperatures(head_key, h.device, proto_scores.dtype)
        pooled_residual = self._pool_class_residual(proto_scores_logits, class_temperatures)
        candidate_mask = torch.ones_like(pooled_residual)
        anchor_residual_path_weights = None
        residual_anchor_support_gate = None
        local_proto_pool_weights = torch.softmax(local_proto_routing_scores / assign_tau, dim=-1)
        local_pooled_residual = torch.sum(local_proto_pool_weights * local_proto_scores_logits, dim=-1)
        local_slot_reliability = None
        candidate_centered = (
            self.residual_logit_mode
            in {
                "local_evidence",
                "local_competition",
                "agreement_candidate_centered",
                "agreement_candidate_margin_mix",
            }
            or (
                candidate_competition
                and self.residual_logit_mode != "agreement_local_centered"
            )
        )
        if self.residual_logit_mode == "local_evidence":
            # Global anchor supplies class semantics; local abnormal evidence only
            # explains residual differences among anchor-plausible candidates.
            proto_residual_scores_raw = local_proto_residual_scores_raw
            proto_scores = local_proto_scores
            proto_scores_logits = local_proto_scores_logits
            proto_routing_scores = local_proto_routing_scores
            pooled_residual = local_pooled_residual
        elif self.residual_logit_mode == "agreement_local_centered":
            # Keep the global anchor as the class-level prior while allowing
            # local abnormal evidence to correct any class through a centered
            # residual. This avoids hard top-k exclusion by the anchor path.
            pooled_residual = local_pooled_residual - local_pooled_residual.mean(
                dim=1,
                keepdim=True,
            )
        elif self.residual_logit_mode in {
            "global_local_consensus",
            "agreement_global_local_consensus",
        }:
            # Single interpretable prototype residual path:
            # global/fused evidence and local abnormal evidence are expressed in
            # the same prototype-offset coordinates and averaged before class
            # pooling. The class anchor remains purely global; local evidence can
            # only explain centered class-internal deviations around that anchor.
            proto_residual_scores_raw = 0.5 * (
                proto_residual_scores_raw + local_proto_residual_scores_raw
            )
            proto_scores = 0.5 * (proto_scores + local_proto_scores)
            proto_scores_logits = 0.5 * (proto_scores_logits + local_proto_scores_logits)
            proto_routing_scores = 0.5 * (proto_routing_scores + local_proto_routing_scores)
            pooled_residual = self._pool_class_residual(proto_scores_logits, class_temperatures)
            pooled_residual = pooled_residual - pooled_residual.mean(
                dim=1,
                keepdim=True,
            )
        elif self.residual_logit_mode == "agreement_local_slot_verify":
            # Local abnormal evidence is not a class classifier. It verifies
            # whether the same within-class prototype slot is supported by the
            # global/fused route and the localized anomaly route.
            if self.num_prototypes_per_class > 1:
                global_slot_probs = torch.softmax(proto_routing_scores / assign_tau, dim=-1)
                local_slot_probs = torch.softmax(local_proto_routing_scores / assign_tau, dim=-1)
                slot_alignment = float(self.num_prototypes_per_class) * (
                    global_slot_probs * local_slot_probs
                ).sum(dim=-1)
                local_slot_reliability = slot_alignment / slot_alignment.mean(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-6)
            else:
                local_slot_reliability = torch.ones_like(pooled_residual)
        if candidate_centered and pooled_residual.shape[1] > 1:
            top_count = min(2, int(pooled_residual.shape[1]))
            top_indices = anchor_scores.topk(k=top_count, dim=1).indices
            candidate_mask = torch.zeros_like(pooled_residual)
            candidate_mask.scatter_(1, top_indices, 1.0)
            candidate_count = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            candidate_mean = (local_pooled_residual * candidate_mask).sum(
                dim=1,
                keepdim=True,
            ) / candidate_count
            pooled_residual = (local_pooled_residual - candidate_mean) * candidate_mask
        residual_weight = (
            pooled_residual.new_tensor(float(self.residual_logit_weight))
            if residual_logit_weight is None
            else residual_logit_weight.to(device=pooled_residual.device, dtype=pooled_residual.dtype)
        )
        base_residual_weight = residual_weight
        pooled_residual, residual_weight = self._calibrate_residual_for_anchor(
            pooled_residual,
            anchor_scores,
            base_residual_weight,
        )
        if self.residual_logit_mode == "agreement_anchor_support" and pooled_residual.shape[1] > 1:
            # The class anchor is the semantic decision coordinate. Prototype
            # residuals are allowed to strengthen that class only when their
            # class-level evidence also supports the anchor's top class; otherwise
            # they remain available for contrastive/prototype explanation but do
            # not override the anchor decision.
            class_count = float(pooled_residual.shape[1])
            anchor_top = anchor_scores.detach().argmax(dim=1, keepdim=True)
            residual_probs = torch.softmax(pooled_residual.detach(), dim=1)
            anchor_support = residual_probs.gather(1, anchor_top)
            uniform_support = 1.0 / class_count
            residual_anchor_support_gate = (
                (anchor_support - uniform_support) / max(1.0 - uniform_support, 1e-6)
            ).clamp(0.0, 1.0)
            residual_weight = residual_weight * residual_anchor_support_gate
        residual_evidence = residual_weight * pooled_residual
        if local_slot_reliability is not None:
            residual_evidence = residual_evidence * local_slot_reliability
        if self.residual_logit_mode in {
            "anchor_residual_margin_mix",
            "agreement_anchor_residual_margin_mix",
            "agreement_candidate_margin_mix",
        }:
            # Interpretability-preserving evidence fusion: the class anchor and
            # the prototype residual remain separate evidence paths.  The final
            # decision is a margin-weighted consensus instead of an unconditional
            # additive correction, which prevents a weak residual path from
            # overriding a stable anchor and vice versa.
            anchor_margin = self._class_margin(anchor_scores)
            residual_margin = self._class_margin(residual_evidence)
            anchor_residual_path_weights = torch.softmax(
                torch.stack([anchor_margin, residual_margin], dim=1),
                dim=1,
            )
            logits = (
                anchor_residual_path_weights[:, 0:1] * anchor_scores
                + anchor_residual_path_weights[:, 1:2] * residual_evidence
                + class_bias.view(1, -1)
            )
        else:
            logits = anchor_scores + residual_evidence + class_bias.view(1, -1)
        anchor_path_weights = None

        if alternative_anchor_h is not None:
            alt_anchor_h_norm = F.normalize(alternative_anchor_h, dim=-1)
            alt_full_anchor_scores_raw = torch.einsum("bd,nd->bn", alt_anchor_h_norm, class_anchor)
            alt_anchor_scores_raw = self._class_anchor_scores(alt_anchor_h_norm, class_anchor)
            alt_anchored_proto_residual_scores_raw = None
            if self.residual_score_mode in {"anchored_offset", "anchored_routing"}:
                alt_anchored_proto_residual_scores_raw = self._anchored_offset_scores(
                    h_norm,
                    alt_anchor_h_norm,
                    proto_norm,
                    class_anchor,
                )
            if (
                self.residual_score_mode == "anchored_offset"
                and alt_anchored_proto_residual_scores_raw is not None
            ):
                alt_proto_residual_scores_raw = alt_anchored_proto_residual_scores_raw
            else:
                alt_proto_residual_scores_raw = proto_scores_raw - alt_full_anchor_scores_raw.unsqueeze(-1)
            if self.num_prototypes_per_class > 1:
                alt_proto_routing_source = (
                    alt_anchored_proto_residual_scores_raw
                    if self.residual_score_mode in {"anchored_offset", "anchored_routing"}
                    and alt_anchored_proto_residual_scores_raw is not None
                    else alt_proto_residual_scores_raw
                )
                alt_proto_routing_mean = alt_proto_routing_source.mean(dim=-1, keepdim=True)
                alt_proto_routing_centered = alt_proto_routing_source - alt_proto_routing_mean
                alt_proto_routing_scale = (
                    alt_proto_routing_centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                )
                alt_proto_routing_scores = alt_proto_routing_centered / alt_proto_routing_scale
            else:
                alt_proto_routing_scores = torch.zeros_like(alt_proto_residual_scores_raw)
            alt_anchor_scores = logit_scale * alt_anchor_scores_raw
            alt_proto_scores = logit_scale * alt_proto_residual_scores_raw
            alt_proto_scores_logits = alt_proto_scores + class_effective_k_bias.unsqueeze(0)
            if self.class_pool_mode == "max":
                alt_pooled_residual = alt_proto_scores_logits.max(dim=-1).values
            elif self.class_pool_mode == "mean":
                alt_pooled_residual = alt_proto_scores_logits.mean(dim=-1)
            else:
                alt_pooled_residual = class_temperatures.view(1, -1) * torch.logsumexp(
                    alt_proto_scores_logits / class_temperatures.view(1, -1, 1),
                    dim=-1,
                )
                alt_pooled_residual = alt_pooled_residual - class_temperatures.view(1, -1) * torch.log(
                    torch.tensor(
                        float(max(self.num_prototypes_per_class, 1)),
                        device=alt_proto_scores.device,
                        dtype=alt_proto_scores.dtype,
                    )
                )
            alt_pooled_residual, alt_residual_weight = self._calibrate_residual_for_anchor(
                alt_pooled_residual,
                alt_anchor_scores,
                base_residual_weight,
            )
            alt_residual_evidence = alt_residual_weight * alt_pooled_residual
            alt_logits = (
                alt_anchor_scores
                + alt_residual_evidence
                + class_bias.view(1, -1)
            )
            if str(anchor_mix_mode) == "mean":
                anchor_path_weights = torch.full(
                    (h.shape[0], 2),
                    0.5,
                    device=h.device,
                    dtype=h.dtype,
                )
            elif str(anchor_mix_mode) == "anchor_margin":
                # The two paths are alternative class-anchor views. Select the
                # better anchor by semantic class margin; prototype residuals
                # should explain intra-class offsets, not decide which anchor
                # path becomes the classifier.
                primary_margin = self._class_margin(anchor_scores)
                alt_margin = self._class_margin(alt_anchor_scores)
                anchor_path_weights = torch.softmax(
                    torch.stack([primary_margin, alt_margin], dim=1),
                    dim=1,
                )
            elif str(anchor_mix_mode) == "residual_margin":
                primary_margin = self._class_margin(pooled_residual)
                alt_margin = self._class_margin(alt_pooled_residual)
                anchor_path_weights = torch.softmax(
                    torch.stack([primary_margin, alt_margin], dim=1),
                    dim=1,
                )
            else:
                # Default dual-residual routing: the global anchor remains the
                # primary semantic frame. Fused evidence is allowed to correct
                # it only when the global anchor is uncertain, and the correction
                # strength is selected by prototype-offset explanation clarity.
                primary_margin = self._class_margin(pooled_residual)
                alt_margin = self._class_margin(alt_pooled_residual)
                residual_path_weights = torch.softmax(
                    torch.stack([primary_margin, alt_margin], dim=1),
                    dim=1,
                )
                if anchor_scores.shape[1] <= 1:
                    global_uncertainty = torch.zeros(
                        h.shape[0],
                        1,
                        device=h.device,
                        dtype=h.dtype,
                    )
                else:
                    anchor_prior = torch.softmax(anchor_scores.detach(), dim=1)
                    class_count = float(anchor_scores.shape[1])
                    max_prior = anchor_prior.max(dim=1, keepdim=True).values
                    global_uncertainty = (
                        (1.0 - max_prior) * class_count / (class_count - 1.0)
                    ).clamp(min=0.0, max=1.0)
                alt_weight = residual_path_weights[:, 1:2] * global_uncertainty
                primary_weight = 1.0 - alt_weight
                anchor_path_weights = torch.cat([primary_weight, alt_weight], dim=1)
            primary_w = anchor_path_weights[:, 0].view(-1, 1)
            alt_w = anchor_path_weights[:, 1].view(-1, 1)
            primary_w_proto = primary_w.unsqueeze(-1)
            alt_w_proto = alt_w.unsqueeze(-1)
            anchor_scores_raw = primary_w * anchor_scores_raw + alt_w * alt_anchor_scores_raw
            anchor_scores = primary_w * anchor_scores + alt_w * alt_anchor_scores
            proto_residual_scores_raw = (
                primary_w_proto * proto_residual_scores_raw
                + alt_w_proto * alt_proto_residual_scores_raw
            )
            proto_scores = primary_w_proto * proto_scores + alt_w_proto * alt_proto_scores
            proto_scores_logits = (
                primary_w_proto * proto_scores_logits
                + alt_w_proto * alt_proto_scores_logits
            )
            proto_routing_scores = (
                primary_w_proto * proto_routing_scores
                + alt_w_proto * alt_proto_routing_scores
            )
            pooled_residual = primary_w * pooled_residual + alt_w * alt_pooled_residual
            if anchor_residual_path_weights is None:
                logits = primary_w * logits + alt_w * alt_logits
            else:
                residual_evidence = primary_w * residual_evidence + alt_w * alt_residual_evidence
                anchor_margin = self._class_margin(anchor_scores)
                residual_margin = self._class_margin(residual_evidence)
                anchor_residual_path_weights = torch.softmax(
                    torch.stack([anchor_margin, residual_margin], dim=1),
                    dim=1,
                )
                logits = (
                    anchor_residual_path_weights[:, 0:1] * anchor_scores
                    + anchor_residual_path_weights[:, 1:2] * residual_evidence
                    + class_bias.view(1, -1)
                )

        if labels is None:
            target_class_ids = logits.argmax(dim=1)
        else:
            target_class_ids = labels.to(h.device).long()

        gather_index = target_class_ids.view(-1, 1, 1).expand(-1, 1, self.num_prototypes_per_class)
        target_proto_scores = proto_scores.gather(1, gather_index).squeeze(1)
        target_proto_scores_raw = proto_residual_scores_raw.gather(1, gather_index).squeeze(1)
        if routing_h is not None:
            # Prototype identity should be supported by both the global concept
            # route and the local abnormal evidence route. Local evidence is a
            # correction signal, not a replacement classifier.
            assignment_routing_scores = 0.5 * (proto_routing_scores + local_proto_routing_scores)
        else:
            assignment_routing_scores = proto_routing_scores
        target_proto_routing_scores = assignment_routing_scores.gather(1, gather_index).squeeze(1)
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
        if labels is not None and update_enabled:
            self._update_anchor_memory(head_key, labels.to(h.device).long(), anchor_h_norm.detach())
        return {
            "logits": logits,
            "anchor_scores": anchor_scores,
            "anchor_scores_raw": anchor_scores_raw,
            "class_anchor": class_anchor,
            "proto_scores": proto_scores,
            "proto_scores_logits": proto_scores_logits,
            "proto_scores_raw": proto_scores_raw,
            "proto_residual_scores_raw": proto_residual_scores_raw,
            "pooled_residual_scores": pooled_residual,
            "proto_routing_scores": proto_routing_scores,
            "local_proto_routing_scores": local_proto_routing_scores,
            "local_proto_pool_weights": local_proto_pool_weights,
            "local_slot_reliability": local_slot_reliability,
            "prototype_candidate_mask": candidate_mask,
            "anchor_path_weights": anchor_path_weights,
            "anchor_residual_path_weights": anchor_residual_path_weights,
            "residual_anchor_support_gate": residual_anchor_support_gate,
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
        _, normalized = self._materialize_anchor_and_prototypes(head_key, prototypes)
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
        _, normalized = self._materialize_anchor_and_prototypes(head_key, prototypes)
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
        _, anchor_memory_initialized = self._get_anchor_memory(head_key)
        sum_counts = counts.sum(dim=1)
        assignment_ratio = torch.where(
            sum_counts > 0,
            counts.max(dim=1).values / sum_counts.clamp_min(1e-6),
            torch.zeros_like(sum_counts),
        )
        probs = counts / sum_counts.unsqueeze(1).clamp_min(1.0)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=1)
        n_eff = torch.where(sum_counts > 0, torch.exp(entropy), torch.zeros_like(sum_counts))
        _, normalized = self._materialize_anchor_and_prototypes(head_key, prototypes.detach())
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
            "class_anchor_memory_weight": float(self.class_anchor_memory_weight),
            "class_anchor_memory_initialized": anchor_memory_initialized.detach().cpu().int().tolist(),
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
