from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_masks import build_feature_mask, describe_active_components
from .config_schema import TFMultiProtoDGConfig, build_model_config
from .cross_evidence_pooling import CrossEvidencePooling
from .diagnostics_state import DiagnosticsState
from .evidence_role_compression import EvidenceRoleCompression
from .feature_schema import build_feature_metadata, flatten_feature_tensors
from .freq_operators import build_freq_operator_bank
from .indicators import compute_freq_indicators, compute_time_indicators
from .local_channel_fusion import LocalChannelFusion
from .losses import compute_minimal_prototype_contrastive
from .partitioning import make_freq_bands, make_time_patches
from .prototype_head import PrototypeHead
from .tensor_ops import compute_partition_width, safe_rfft_amplitude, to_bcl
from .time_operators import build_time_operator_bank


class Model(nn.Module):
    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.metadata = metadata
        self.args = args
        self.config: TFMultiProtoDGConfig = build_model_config(args, metadata)

        self.time_operator_bank = build_time_operator_bank(self.config)
        self.freq_operator_bank = build_freq_operator_bank(self.config)
        self.time_patch_width = compute_partition_width(
            self.config.input_length,
            self.config.time_patch_count,
            self.config.padding_mode,
        )
        freq_bins = self.config.input_length // 2 + 1
        self.freq_band_width = compute_partition_width(
            freq_bins,
            self.config.freq_band_count,
            self.config.padding_mode,
        )
        self.feature_meta = build_feature_metadata(
            num_channels=self.config.in_channels,
            time_operator_names=self.time_operator_bank.operator_names,
            freq_operator_names=self.freq_operator_bank.operator_names,
            time_patch_count=self.config.time_patch_count,
            freq_band_count=self.config.freq_band_count,
            time_patch_width=self.time_patch_width,
            freq_band_width=self.freq_band_width,
            time_indicator_names=self.config.time_indicators,
            freq_indicator_names=self.config.freq_indicators,
        )
        self.feature_dim = len(self.feature_meta)
        feature_mask = build_feature_mask(self.feature_meta, self.config.ablation)
        self.register_buffer("feature_mask", feature_mask, persistent=False)
        self.active_component_summary = describe_active_components(self.feature_meta, feature_mask)

        self.time_basis_dim = (
            len(self.time_operator_bank.operator_names) * len(self.config.time_indicators)
        )
        self.freq_basis_dim = (
            len(self.freq_operator_bank.operator_names) * len(self.config.freq_indicators)
        )

        self.time_channel_fusion = LocalChannelFusion(
            in_features=self.time_basis_dim,
            num_channels=self.config.in_channels,
            norm=self.config.channel_fusion_norm,
        )
        self.freq_channel_fusion = LocalChannelFusion(
            in_features=self.freq_basis_dim,
            num_channels=self.config.in_channels,
            norm=self.config.channel_fusion_norm,
        )
        self.time_role_compression = EvidenceRoleCompression(
            in_features=self.time_basis_dim,
            role_dim=self.config.role_dim,
            nonneg=self.config.role_nonneg,
            input_norm=self.config.role_input_norm,
            output_norm=self.config.role_output_norm,
        )
        self.freq_role_compression = EvidenceRoleCompression(
            in_features=self.freq_basis_dim,
            role_dim=self.config.role_dim,
            nonneg=self.config.role_nonneg,
            input_norm=self.config.role_input_norm,
            output_norm=self.config.role_output_norm,
        )
        self.cross_evidence_pooling = CrossEvidencePooling(
            score_norm=self.config.cross_score_norm,
            pool_mode=self.config.cross_pool_mode,
            pool_tau=self.config.cross_pool_tau,
            concept_norm=self.config.concept_norm,
            self_mix=self.config.cross_self_mix,
            self_score_mode=self.config.cross_self_score_mode,
            adaptive_self_mix=self.config.cross_adaptive_self_mix,
        )
        self.raw_feature_residual_norm = nn.LayerNorm(self.feature_dim)
        self.raw_feature_residual_proj = nn.Linear(self.feature_dim, self.config.concept_dim, bias=False)
        self.prototype_head = PrototypeHead(
            num_classes=self.config.num_classes,
            concept_dim=self.config.concept_dim,
            num_prototypes_per_class=self.config.num_prototypes_per_class,
            temperature=self.config.prototype_temperature,
            class_pool_mode=self.config.prototype_class_pool_mode,
            adaptive_class_temperature_enabled=self.config.adaptive_class_temperature_enabled,
            adaptive_class_temperature_target_neff=self.config.adaptive_class_temperature_target_neff,
            adaptive_class_temperature_min_scale=self.config.adaptive_class_temperature_min_scale,
            adaptive_class_temperature_max_scale=self.config.adaptive_class_temperature_max_scale,
            adaptive_effective_k_enabled=self.config.adaptive_effective_k_enabled,
            adaptive_effective_k_ready_count=self.config.adaptive_effective_k_ready_count,
            adaptive_effective_k_min=self.config.adaptive_effective_k_min,
            adaptive_effective_k_penalty=self.config.adaptive_effective_k_penalty,
            assignment_temperature=self.config.prototype_assignment_temperature,
            logit_scale_init=self.config.prototype_logit_scale_init,
        )
        # Compatibility alias for legacy tests and callback assumptions.
        self.prototype_bank = self.prototype_head

        self.export_diagnostics = self.config.diagnostics_enabled
        self.diagnostics_state = DiagnosticsState(
            feature_meta=self.feature_meta,
            time_operator_names=list(self.time_operator_bank.operator_names),
            freq_operator_names=list(self.freq_operator_bank.operator_names),
            time_indicator_names=list(self.config.time_indicators),
            freq_indicator_names=list(self.config.freq_indicators),
            role_dim=self.config.role_dim,
            top_t=self.config.prototype_card_top_t,
            max_members=self.config.diagnostics_max_members,
        )

    def _feature_mask_on(self, device: torch.device) -> torch.Tensor:
        return self.feature_mask.to(device=device)

    @staticmethod
    def _to_tensor_ids(file_ids_raw: Any, device: torch.device) -> torch.Tensor:
        if file_ids_raw is None:
            return torch.zeros(1, device=device, dtype=torch.long)
        if torch.is_tensor(file_ids_raw):
            return file_ids_raw.to(device).long()
        try:
            return torch.as_tensor(file_ids_raw, device=device, dtype=torch.long)
        except Exception:
            return torch.zeros(1, device=device, dtype=torch.long)

    def resolve_head_key(self, file_ids: Optional[torch.Tensor]) -> str:
        if len(self.prototype_head.head_keys) == 1:
            return self.prototype_head.head_keys[0]
        dataset_id = None
        if file_ids is not None and self.metadata is not None and file_ids.numel() > 0:
            try:
                dataset_id = str(self.metadata[int(file_ids.view(-1)[0].item())].get("Dataset_id"))
            except Exception:
                dataset_id = None
        if dataset_id and dataset_id in self.prototype_head.head_keys:
            return dataset_id
        return sorted(self.prototype_head.head_keys)[0]

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

    def _split_feature_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_count = (
            self.config.in_channels
            * len(self.time_operator_bank.operator_names)
            * self.config.time_patch_count
            * len(self.config.time_indicators)
        )
        time_mask = self.feature_mask[:time_count].view(
            self.config.in_channels,
            len(self.time_operator_bank.operator_names),
            self.config.time_patch_count,
            len(self.config.time_indicators),
        )
        freq_mask = self.feature_mask[time_count:].view(
            self.config.in_channels,
            len(self.freq_operator_bank.operator_names),
            self.config.freq_band_count,
            len(self.config.freq_indicators),
        )
        return time_mask, freq_mask

    def _concept_with_residual(
        self,
        h_core: torch.Tensor,
        h_raw_full: torch.Tensor,
        h_raw_struct_masked: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_weight = float(self.config.classifier.residual_weight)
        if residual_weight <= 0.0:
            return h_core, h_core.new_zeros(h_core.shape)

        residual_input = str(self.config.classifier.residual_input)
        if residual_input == "raw_feature":
            residual_source = h_raw_full
        elif residual_input == "structured_topk":
            residual_source = h_raw_struct_masked
            topk = self.config.classifier.residual_topk
            if topk is None:
                topk = self.config.top_k_features if self.config.top_k_features is not None else 64
            k = min(int(topk), residual_source.shape[-1])
            if k > 0 and k < residual_source.shape[-1]:
                topk_idx = residual_source.abs().topk(k=k, dim=-1).indices
                topk_mask = torch.zeros_like(residual_source).scatter(1, topk_idx, 1.0)
                residual_source = residual_source * topk_mask
        else:
            residual_source = h_raw_struct_masked
        residual = self.raw_feature_residual_proj(self.raw_feature_residual_norm(residual_source))
        residual = float(self.config.classifier.scale) * residual
        return h_core + residual_weight * residual, residual

    def _extract_local_feature_tensors(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_bcl = to_bcl(x)
        time_stack = self.time_operator_bank(x_bcl)
        freq_stack = self.freq_operator_bank(safe_rfft_amplitude(x_bcl))
        time_patches = make_time_patches(time_stack, self.config.time_patch_count, self.config.padding_mode)
        freq_bands = make_freq_bands(freq_stack, self.config.freq_band_count, self.config.padding_mode)
        time_features = compute_time_indicators(time_patches, self.config.time_indicators)
        freq_features = compute_freq_indicators(freq_bands, self.config.freq_indicators)

        time_mask, freq_mask = self._split_feature_mask()
        time_features_masked = time_features * time_mask.to(time_features.device).unsqueeze(0)
        freq_features_masked = freq_features * freq_mask.to(freq_features.device).unsqueeze(0)

        h_raw_full = flatten_feature_tensors(time_features, freq_features)
        h_raw_struct_masked = flatten_feature_tensors(time_features_masked, freq_features_masked)

        e_t_flat = (
            time_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, self.config.time_patch_count, self.time_basis_dim)
        )
        e_f_flat = (
            freq_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, self.config.freq_band_count, self.freq_basis_dim)
        )
        return h_raw_full, h_raw_struct_masked, time_features_masked, freq_features_masked, e_t_flat, e_f_flat

    @staticmethod
    def _zero_contrastive_dict(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "total": zero,
            "lambda_schedule": zero,
            "adaptive_proto_scale": zero,
            "proto_nce": zero,
            "prototype_contrastive": zero,
            "balance_penalty": zero,
            "occupancy_penalty": zero,
            "assignment_ratio": zero,
            "specialization_penalty": zero,
            "specialization_gap": zero,
            "specialization_competitor_score": zero,
            "readability_penalty": zero,
            "complementarity_penalty": zero,
        }

    def _scheduled_proto_weight(self, epoch: Optional[int], total_epochs: Optional[int]) -> float:
        warmup_epochs = max(int(self.config.contrastive_warmup_epochs), 0)
        current_epoch = max(int(epoch or 0), 0)
        if current_epoch < warmup_epochs:
            return 0.0
        if self.config.proto_contrastive_weight > 0.0:
            return float(self.config.proto_contrastive_weight)
        if total_epochs is None or total_epochs <= 1:
            ratio = 1.0
        else:
            remaining_epochs = max(int(total_epochs) - warmup_epochs - 1, 1)
            progressed = max(current_epoch - warmup_epochs, 0)
            ratio = min(max(float(progressed) / float(remaining_epochs), 0.0), 1.0)
        return float(
            self.config.lambda_cl_start + (self.config.lambda_cl_end - self.config.lambda_cl_start) * ratio
        )

    def _adaptive_proto_scale(self, contrastive: Dict[str, torch.Tensor]) -> torch.Tensor:
        reference = contrastive["total"]
        scale = reference.new_tensor(1.0)

        if self.config.adaptive_proto_enabled:
            assignment_ratio = contrastive.get("assignment_ratio", reference.new_zeros(()))
            balance_penalty = contrastive.get("balance_penalty", reference.new_zeros(()))
            assign_denom = max(1.0 - float(self.config.adaptive_proto_assignment_target), 1e-6)
            balance_target = max(float(self.config.adaptive_proto_balance_target), 1e-6)

            assign_signal = torch.relu(
                (assignment_ratio - float(self.config.adaptive_proto_assignment_target)) / assign_denom
            )
            balance_signal = torch.relu(
                (balance_penalty - float(self.config.adaptive_proto_balance_target)) / balance_target
            )
            mix = float(self.config.adaptive_proto_balance_mix)
            signal = ((1.0 - mix) * assign_signal + mix * balance_signal).clamp(0.0, 1.0)
            min_scale = float(self.config.adaptive_proto_min_scale)
            max_scale = float(self.config.adaptive_proto_max_scale)
            scale = scale * (reference.new_tensor(min_scale) + (max_scale - min_scale) * signal)

        return scale

    def _forward_structured(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        (
            h_raw_full,
            h_raw_struct_masked,
            e_t,
            e_f,
            e_t_flat,
            e_f_flat,
        ) = self._extract_local_feature_tensors(x)
        file_ids = self._to_tensor_ids(file_ids_raw, h_raw_full.device)
        domains = self.get_domains(file_ids)
        head_key = self.resolve_head_key(file_ids)

        e_t_bar, alpha_t = self.time_channel_fusion(e_t_flat)
        e_f_bar, alpha_f = self.freq_channel_fusion(e_f_flat)
        z_t, a_t = self.time_role_compression(e_t_bar)
        z_f, a_f = self.freq_role_compression(e_f_bar)
        h_core, s, w_t, w_f, g_t, g_f = self.cross_evidence_pooling(z_t, z_f)
        h, h_residual = self._concept_with_residual(h_core, h_raw_full, h_raw_struct_masked)
        proto_out = self.prototype_head(h, labels=labels, head_key=head_key)

        extras: Dict[str, Any] = {
            "logits": proto_out["logits"],
            "E_t": e_t,
            "E_f": e_f,
            "E_t_flat": e_t_flat,
            "E_f_flat": e_f_flat,
            "alpha_t": alpha_t,
            "alpha_f": alpha_f,
            "E_t_bar": e_t_bar,
            "E_f_bar": e_f_bar,
            "A_t": a_t,
            "A_f": a_f,
            "Z_t": z_t,
            "Z_f": z_f,
            "S": s,
            "w_t": w_t,
            "w_f": w_f,
            "g_t": g_t,
            "g_f": g_f,
            "h_core": h_core,
            "h_residual": h_residual,
            "h": h,
            "c": h,
            "proto_scores": proto_out["proto_scores"],
            "proto_scores_logits": proto_out["proto_scores_logits"],
            "proto_scores_raw": proto_out["proto_scores_raw"],
            "target_class_ids": proto_out["target_class_ids"],
            "class_neff": proto_out["class_neff"],
            "target_class_neff": proto_out["target_class_neff"],
            "class_temperatures": proto_out["class_temperatures"],
            "target_class_temperatures": proto_out["target_class_temperatures"],
            "class_effective_k_bias": proto_out["class_effective_k_bias"],
            "class_effective_k": proto_out["class_effective_k"],
            "target_class_effective_k": proto_out["target_class_effective_k"],
            "target_proto_scores": proto_out["target_proto_scores"],
            "target_proto_scores_routed": proto_out["target_proto_scores_routed"],
            "target_proto_probs": proto_out["target_proto_probs"],
            "prototype_assignments": proto_out["prototype_assignments"],
            "prototype_assignment_weights": proto_out["prototype_assignment_weights"],
            "prototype_pos_scores": proto_out["prototype_pos_scores"],
            "file_ids": file_ids,
            "domains": domains,
            "head_key": head_key,
            "labels": labels,
            "h_raw_full": h_raw_full,
            "h_raw_struct_masked": h_raw_struct_masked,
            "h_raw": h_raw_struct_masked,
            "h_masked": h_raw_struct_masked,
            "topk_idx": None,
            "topk_mask": None,
            "variant_id": self.config.variant_id,
            "active_components": self.active_component_summary,
            "concept_source": "lsep_simplified",
        }
        return extras

    def forward_with_batch(self, batch: Dict[str, Any], epoch: int = 0) -> Dict[str, Any]:
        labels = batch.get("y")
        labels_tensor = labels.to(self.feature_mask.device).long() if torch.is_tensor(labels) else None
        return self._forward_structured(
            x=batch["x"],
            file_ids_raw=batch.get("_file_ids_raw", batch.get("file_id")),
            labels=labels_tensor,
        )

    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        extras = self._forward_structured(x=x, file_ids_raw=file_id, labels=None)
        return extras["logits"]

    def compute_contrastive_loss(
        self,
        extras: Dict[str, Any],
        labels: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.config.use_contrastive_head:
            return self._zero_contrastive_dict(extras["h"])

        scheduled_proto_weight = self._scheduled_proto_weight(epoch, total_epochs)
        warmup_epochs = max(int(self.config.contrastive_warmup_epochs), 0)
        current_epoch = max(int(epoch or 0), 0)
        warmup_gate = 0.0 if current_epoch < warmup_epochs else 1.0
        proto_weight = scheduled_proto_weight * self.config.contrastive_scale
        specialization_weight = warmup_gate * float(self.config.proto_specialization_weight)
        diversity_weight = float(self.config.prototype_diversity_weight)
        occupancy_weight = float(self.config.prototype_occupancy_weight)
        if (
            proto_weight <= 0.0
            and specialization_weight <= 0.0
            and diversity_weight <= 0.0
            and occupancy_weight <= 0.0
        ):
            zero = self._zero_contrastive_dict(extras["h"])
            zero["lambda_schedule"] = extras["h"].new_tensor(0.0)
            return zero

        total = extras["h"].new_zeros(())
        if (
            proto_weight > 0.0
            or specialization_weight > 0.0
            or occupancy_weight > 0.0
        ):
            target_proto_scores = extras.get("target_proto_scores_routed", extras.get("target_proto_scores"))
            target_proto_probs = extras.get("target_proto_probs")
            if self.config.proto_contrastive_input == "h_core":
                head_key = extras["head_key"]
                prototypes, logit_scale_raw, _, _, _, _ = self.prototype_head._get_buffers(head_key)
                core_concepts = F.normalize(extras["h_core"], dim=-1)
                proto_norm = F.normalize(prototypes, dim=-1)
                contrastive_scores_raw = torch.einsum("bd,nkd->bnk", core_concepts, proto_norm)
                if self.config.proto_contrastive_use_scaled_scores:
                    logit_scale = F.softplus(logit_scale_raw) + 1e-4
                    contrastive_scores = logit_scale * contrastive_scores_raw
                else:
                    contrastive_scores = contrastive_scores_raw
                target_proto_scores = contrastive_scores[
                    torch.arange(labels.shape[0], device=labels.device),
                    labels.to(contrastive_scores.device).long(),
                ]
                assign_tau = max(float(self.config.prototype_assignment_temperature), 1e-6)
                target_proto_probs = torch.softmax(target_proto_scores / assign_tau, dim=-1)
            else:
                if self.config.proto_contrastive_use_scaled_scores:
                    contrastive_scores = extras["proto_scores"]
                else:
                    contrastive_scores = extras.get("proto_scores_raw", extras["proto_scores"])
            contrastive = compute_minimal_prototype_contrastive(
                proto_scores=contrastive_scores,
                labels=labels,
                temperature=self.config.proto_contrastive_temperature,
                assignment_temperature=self.config.prototype_assignment_temperature,
                neg_topk=self.config.proto_contrastive_neg_topk,
                positive_mode=self.config.proto_contrastive_positive_mode,
                specialization_margin=self.config.proto_specialization_margin,
                specialization_temperature=self.config.proto_specialization_temperature,
                domains=extras.get("domains"),
                target_proto_scores=target_proto_scores,
                target_proto_probs=target_proto_probs,
                target_class_neff=(
                    extras.get("target_class_neff")
                    if self.config.adaptive_contrastive_neff_enabled
                    else None
                ),
                adaptive_neff_min_scale=(
                    float(self.config.adaptive_contrastive_neff_min_scale)
                    if self.config.adaptive_contrastive_neff_enabled
                    else 1.0
                ),
                adaptive_neff_max_scale=(
                    float(self.config.adaptive_contrastive_neff_max_scale)
                    if self.config.adaptive_contrastive_neff_enabled
                    else 1.0
                ),
            )
            extras.update(
                {
                    "target_proto_probs": contrastive["target_proto_probs"],
                    "proto_positive_scores": contrastive["proto_positive_scores"],
                    "proto_negative_scores": contrastive["proto_negative_scores"],
                    "prototype_usage": contrastive["prototype_usage"],
                }
            )
        else:
            contrastive = self._zero_contrastive_dict(extras["h"])
            contrastive["n_eff"] = extras["h"].new_zeros(1)
        adaptive_proto_scale = self._adaptive_proto_scale(contrastive)
        effective_proto_weight = extras["h"].new_tensor(proto_weight) * adaptive_proto_scale
        contrastive["adaptive_proto_scale"] = adaptive_proto_scale
        effective_specialization_weight = extras["h"].new_tensor(specialization_weight)

        if diversity_weight > 0.0:
            diversity_penalty = self.prototype_head.diversity_penalty(extras["head_key"])
            diversity_scale = extras["h"].new_tensor(1.0)
            adaptive_gain = float(self.config.prototype_diversity_adaptive_gain)
            if adaptive_gain > 0.0:
                diversity_excess = self.prototype_head.diversity_excess(
                    extras["head_key"],
                    target_cos=float(self.config.prototype_diversity_target_cos),
                )
                diversity_scale = 1.0 + adaptive_gain * diversity_excess.detach()
        else:
            diversity_penalty = extras["h"].new_zeros(())
            diversity_scale = extras["h"].new_tensor(1.0)
        contrastive["prototype_diversity_penalty"] = diversity_penalty
        contrastive["prototype_diversity_scale"] = diversity_scale
        total = (
            total
            + effective_proto_weight * contrastive["total"]
            + effective_specialization_weight * contrastive["specialization_penalty"]
            + diversity_weight * diversity_scale * diversity_penalty
            + occupancy_weight * contrastive["occupancy_penalty"]
        )
        contrastive["total"] = total
        # Default_task multiplies dict-based contrastive outputs by lambda_schedule once more.
        # Return the fully weighted loss here and keep lambda_schedule at 1.0 to avoid
        # suppressing prototype contrastive updates by applying the schedule twice.
        contrastive["proto_lambda_schedule"] = effective_proto_weight
        contrastive["specialization_lambda_schedule"] = effective_specialization_weight
        contrastive["lambda_schedule"] = extras["h"].new_tensor(1.0)
        return contrastive

    def update_diagnostics(self, batch: Dict[str, Any], extras: Dict[str, Any], stage: str) -> None:
        if not self.export_diagnostics:
            return
        self.diagnostics_state.update(stage, extras)

    def reset_diagnostics(self, stage: str) -> None:
        self.diagnostics_state.reset_stage(stage)

    def export_diagnostics_payload(self, stage: str, epoch: int) -> Dict[str, object]:
        if not self.export_diagnostics:
            return {}
        payload = self.diagnostics_state.build_stage_payload(
            stage=stage,
            prototype_head=self.prototype_head,
            variant_id=self.config.variant_id,
            active_components=self.active_component_summary,
        )
        payload["epoch"] = int(epoch)
        payload["stage"] = stage
        return payload
