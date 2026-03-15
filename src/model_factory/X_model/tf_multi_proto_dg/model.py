from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .ablation_masks import build_feature_mask, describe_active_components
from .config_schema import TFMultiProtoDGConfig, build_model_config
from .diagnostics_state import DiagnosticsState
from .feature_encoder import FeatureEncoder
from .feature_schema import build_feature_metadata, flatten_feature_tensors
from .freq_operators import build_freq_operator_bank
from .indicators import compute_freq_indicators, compute_time_indicators
from .partitioning import make_freq_bands, make_time_patches
from .prototype_bank import PrototypeBank
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
            self.config.input_length, self.config.time_patch_count, self.config.padding_mode
        )
        freq_bins = self.config.input_length // 2 + 1
        self.freq_band_width = compute_partition_width(
            freq_bins, self.config.freq_band_count, self.config.padding_mode
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
        if self.config.top_k_features is not None and self.config.top_k_features > self.feature_dim:
            raise ValueError(
                f"model.top_k_features={self.config.top_k_features} exceeds feature_dim={self.feature_dim}."
            )
        feature_mask = build_feature_mask(self.feature_meta, self.config.ablation)
        self.register_buffer("feature_mask", feature_mask, persistent=False)
        self.active_component_summary = describe_active_components(self.feature_meta, feature_mask)

        self.encoder = FeatureEncoder(
            feature_dim=self.feature_dim,
            concept_dim=self.config.concept_dim,
            num_classes=self.config.num_classes,
            metadata=metadata,
            use_topk_selector=self.config.use_topk_selector,
            top_k_features=self.config.top_k_features,
            score_mode=self.config.topk_score_mode,
            ema_momentum=self.config.topk_ema_momentum,
            warmup_epochs=self.config.topk_warmup_epochs,
        )
        self.prototype_bank = PrototypeBank(
            num_classes=self.config.num_classes,
            concept_dim=self.config.concept_dim,
            num_prototypes_per_class=self.config.num_prototypes_per_class,
            ema_momentum=self.config.proto_ema_momentum,
            neg_k=self.config.proto_neg_k,
            empty_reset_steps=self.config.proto_empty_reset_steps,
        )
        self.export_diagnostics = self.config.diagnostics_enabled
        self.diagnostics_state = DiagnosticsState(
            feature_meta=self.feature_meta,
            top_t=self.config.prototype_card_top_t,
            max_members=self.config.diagnostics_max_members,
        )

    @property
    def metric(self) -> nn.Linear:
        return self.encoder.metric

    def _feature_mask_on(self, device: torch.device) -> torch.Tensor:
        return self.feature_mask.to(device=device)

    def _apply_structural_mask(self, h_raw: torch.Tensor) -> torch.Tensor:
        return h_raw * self._feature_mask_on(h_raw.device)

    def _masked_metric_weight(self) -> torch.Tensor:
        return self.metric.weight * self._feature_mask_on(self.metric.weight.device).unsqueeze(0)

    @staticmethod
    def _zero_contrastive_dict(reference: torch.Tensor, assignment_ratio: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        zero = reference.new_zeros(())
        return {
            "total": zero,
            "lambda_schedule": zero,
            "proto_nce": zero,
            "readability_penalty": zero,
            "complementarity_penalty": zero,
            "assignment_ratio": assignment_ratio if assignment_ratio is not None else zero,
        }

    def _assignment_ratio(self, assignments: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        if assignments is None:
            return reference.new_zeros(())
        bincount = torch.bincount(assignments.detach().view(-1), minlength=self.config.num_prototypes_per_class).float()
        return (bincount.max() / bincount.sum().clamp_min(1.0)).to(reference.device)

    def _extract_interpretable_features(self, x: torch.Tensor) -> torch.Tensor:
        x_bcl = to_bcl(x)
        time_stack = self.time_operator_bank(x_bcl)
        freq_stack = self.freq_operator_bank(safe_rfft_amplitude(x_bcl))
        time_patches = make_time_patches(time_stack, self.config.time_patch_count, self.config.padding_mode)
        freq_bands = make_freq_bands(freq_stack, self.config.freq_band_count, self.config.padding_mode)
        time_features = compute_time_indicators(time_patches, self.config.time_indicators)
        freq_features = compute_freq_indicators(freq_bands, self.config.freq_indicators)
        return flatten_feature_tensors(time_features, freq_features)

    def forward_with_batch(self, batch: Dict[str, Any], epoch: int = 0) -> Dict[str, Any]:
        h_raw_full = self._extract_interpretable_features(batch["x"])
        h_raw = self._apply_structural_mask(h_raw_full)
        labels = batch.get("y")
        labels_tensor = labels.to(h_raw.device).long() if torch.is_tensor(labels) else None
        extras = self.encoder.encode(
            h_raw=h_raw,
            file_ids_raw=batch.get("_file_ids_raw", batch.get("file_id")),
            labels=labels_tensor,
            epoch=epoch,
        )
        extras["labels"] = labels_tensor
        extras["h_raw_full"] = h_raw_full
        extras["h_raw_struct_masked"] = h_raw
        extras["variant_id"] = self.config.variant_id
        extras["active_components"] = self.active_component_summary

        if labels_tensor is not None and self.config.use_contrastive_head and self.config.prototype_assignment_enabled:
            assignments = self.prototype_bank.assign(extras["c"], labels_tensor, str(extras["head_key"]))
            extras["prototype_assignments"] = assignments["assignments"]
            extras["prototype_pos_scores"] = assignments["pos_scores"]
        else:
            extras["prototype_assignments"] = None
            extras["prototype_pos_scores"] = None
        return extras

    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        h_raw_full = self._extract_interpretable_features(x)
        extras = self.encoder.encode(
            h_raw=self._apply_structural_mask(h_raw_full),
            file_ids_raw=file_id,
            labels=None,
            epoch=0,
        )
        return extras["logits"]

    def compute_contrastive_loss(
        self,
        extras: Dict[str, Any],
        labels: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assignments = extras.get("prototype_assignments")
        assignment_ratio = self._assignment_ratio(assignments, extras["c"])
        if not self.config.use_contrastive_head or assignments is None:
            return self._zero_contrastive_dict(extras["c"], assignment_ratio=assignment_ratio)

        if total_epochs is None or total_epochs <= 1:
            ratio = 1.0
        else:
            ratio = min(max(float(epoch or 0) / float(total_epochs - 1), 0.0), 1.0)
        lambda_schedule = self.config.lambda_cl_start + (
            self.config.lambda_cl_end - self.config.lambda_cl_start
        ) * ratio
        lambda_schedule *= self.config.contrastive_scale

        labels_device = labels.to(extras["c"].device).long()
        head_key = str(extras["head_key"])
        if self.config.prototype_update_enabled:
            self.prototype_bank.update(
                c=extras["c"].detach(),
                labels=labels_device,
                assignments=assignments.detach(),
                head_key=head_key,
            )
        if lambda_schedule <= 0.0:
            losses = self._zero_contrastive_dict(extras["c"], assignment_ratio=assignment_ratio)
            losses["lambda_schedule"] = extras["c"].new_tensor(lambda_schedule)
            return losses

        losses = self.prototype_bank.compute_losses(
            c=extras["c"],
            labels=labels_device,
            assignments=assignments,
            head_key=head_key,
            metric_weight=self._masked_metric_weight(),
            temperature=self.config.temperature,
            readability_weight=self.config.readability_weight * self.config.ablation.loss_control.readability_scale,
            complementarity_weight=self.config.complementarity_weight * self.config.ablation.loss_control.complementarity_scale,
        )
        losses["lambda_schedule"] = extras["c"].new_tensor(lambda_schedule)
        return losses

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
            prototype_bank=self.prototype_bank if self.config.prototype_assignment_enabled else None,
            metric_weight=self._masked_metric_weight().detach(),
            feature_mask=self.feature_mask.detach(),
            variant_id=self.config.variant_id,
            active_components=self.active_component_summary,
        )
        payload["epoch"] = int(epoch)
        payload["stage"] = stage
        return payload
