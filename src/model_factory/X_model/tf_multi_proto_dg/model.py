from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_masks import build_feature_mask, describe_active_components
from .cooperative_prototype_fusion import CooperativePrototypeFusion
from .config_schema import TFMultiProtoDGConfig, build_model_config
from .cross_evidence_pooling import CrossEvidencePooling
from .diagnostics_state import DiagnosticsState
from .evidence_role_compression import EvidenceRoleCompression
from .feature_schema import build_feature_metadata, flatten_feature_tensors
from .freq_operators import build_freq_operator_bank
from .hierarchical_evidence_integration import HierarchicalEvidenceIntegration
from .indicators import compute_freq_indicators, compute_time_indicators
from .local_channel_fusion import LocalChannelFusion
from .losses import compute_minimal_prototype_contrastive
from .partitioning import (
    make_freq_bands,
    make_time_patches,
    resolve_freq_band_width,
    resolve_time_patch_width,
)
from .prototype_head import PrototypeHead
from .tensor_ops import safe_rfft_amplitude, to_bcl
from .time_operators import build_time_operator_bank
from .transparent_global_backbone import TransparentGlobalBackbone


class Model(nn.Module):
    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.metadata = metadata
        self.args = args
        self.config: TFMultiProtoDGConfig = build_model_config(args, metadata)

        self.time_operator_bank = build_time_operator_bank(self.config)
        self.freq_operator_bank = build_freq_operator_bank(self.config)
        self.time_patch_width = resolve_time_patch_width(
            length=self.config.input_length,
            patch_count=self.config.time_patch_count,
            padding_mode=self.config.padding_mode,
            patch_width=self.config.time_patch_width,
        )
        # Multi-granularity: resolve all secondary widths.
        if self.config.time_patch_widths is not None:
            self.time_patch_widths = [
                resolve_time_patch_width(
                    length=self.config.input_length,
                    patch_count=self.config.time_patch_count,
                    padding_mode=self.config.padding_mode,
                    patch_width=w,
                )
                for w in self.config.time_patch_widths
            ]
        else:
            self.time_patch_widths = None
        freq_bins = self.config.input_length // 2 + 1
        self.freq_band_width = resolve_freq_band_width(
            freq_bins=freq_bins,
            band_count=self.config.freq_band_count,
            padding_mode=self.config.padding_mode,
            band_width=self.config.freq_band_width,
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
        self.active_component_summary["time_patch_mode"] = str(self.config.time_patch_mode)
        self.active_component_summary["freq_band_mode"] = str(self.config.freq_band_mode)
        self.active_component_summary["region_sampling_mode"] = str(self.config.region_sampling_mode)
        self.active_component_summary["region_sampling_seed_offset"] = int(
            self.config.region_sampling_seed_offset
        )
        self.active_component_summary["region_ensemble_samples"] = int(self.config.region_ensemble_samples)
        if self.time_patch_widths is not None:
            self.active_component_summary["multi_granularity_time"] = {
                "widths": self.time_patch_widths,
                "mode": "role_average",
            }
        self.active_component_summary["hierarchical_evidence"] = {
            "enabled": True,
            "global_anchor": "dual_deep_global_tf_summary",
            "structured_global_enabled": bool(self.config.structured_global_enabled),
            "global_fusion_mode": (
                "agreement_adaptive_deep_structured"
                if bool(self.config.structured_global_enabled)
                else "deep_only"
            ),
            "branch_summaries": ["time_summary", "freq_summary"],
        }
        mix_total = (
            float(self.config.cooperative_evidence_mix)
            + float(self.config.cooperative_confidence_mix)
            + float(self.config.cooperative_agreement_mix)
        )
        self.cooperative_enabled = bool(self.config.cooperative_prototypes_enabled)
        self.active_component_summary["cooperative_prototypes"] = {
            "enabled": self.cooperative_enabled,
            "heads": ["time", "freq", "joint"] if self.cooperative_enabled else ["joint"],
            "priors": {
                "time": float(self.config.cooperative_time_prior),
                "freq": float(self.config.cooperative_freq_prior),
                "joint": float(self.config.cooperative_joint_prior),
            },
            "fusion_mix": {
                "evidence": float(self.config.cooperative_evidence_mix) / mix_total,
                "confidence": float(self.config.cooperative_confidence_mix) / mix_total,
                "agreement": float(self.config.cooperative_agreement_mix) / mix_total,
            },
            "weight_floor": float(self.config.cooperative_weight_floor),
            "max_share": float(self.config.cooperative_max_share),
            "logit_scale_min": float(self.config.cooperative_logit_scale_min),
        }
        self.active_component_summary["prototype_structure"] = {
            "classifier": "class_anchor_plus_residual_prototypes",
            "num_prototypes_per_class": int(self.config.num_prototypes_per_class),
            "assignment_space": "standardized_prototype_residual",
            "contrastive_space": (
                "scaled_classifier_scores"
                if bool(self.config.proto_contrastive_use_scaled_scores)
                else "standardized_prototype_residual"
            ),
        }
        self.time_basis_dim = (
            len(self.time_operator_bank.operator_names) * len(self.config.time_indicators)
        )
        self.freq_basis_dim = (
            len(self.freq_operator_bank.operator_names) * len(self.config.freq_indicators)
        )
        self.time_summary_topk = self._summary_topk_count(int(self.config.time_patch_count))
        self.freq_summary_topk = self._summary_topk_count(int(self.config.freq_band_count))
        self.global_feature_dim = int(self.config.in_channels * (self.time_basis_dim + self.freq_basis_dim))
        self.semantic_stats_dim = int(
            self.config.in_channels * (self.time_basis_dim + self.freq_basis_dim) * 4
        )
        self.time_local_anomaly_summary_dim = int(self.config.in_channels * self.time_basis_dim * 2)
        self.freq_local_anomaly_summary_dim = int(self.config.in_channels * self.freq_basis_dim * 2)
        self.local_anomaly_summary_dim = int(
            self.time_local_anomaly_summary_dim + self.freq_local_anomaly_summary_dim
        )
        self.time_local_anomaly_profile_dim = int(
            self.config.in_channels * self.time_basis_dim * (1 + self.time_summary_topk)
        )
        self.freq_local_anomaly_profile_dim = int(
            self.config.in_channels * self.freq_basis_dim * (1 + self.freq_summary_topk)
        )
        self.local_anomaly_profile_dim = int(
            self.time_local_anomaly_profile_dim + self.freq_local_anomaly_profile_dim
        )
        self.semantic_coord_stats_dim = 4
        self.semantic_coord_time_token_count = int(self.config.in_channels * self.time_basis_dim)
        self.semantic_coord_token_count = int(
            self.config.in_channels * (self.time_basis_dim + self.freq_basis_dim)
        )
        self.semantic_coord_names = self._build_semantic_coord_names()

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
            init_mode=self.config.role_init_mode,
        )
        self.freq_role_compression = EvidenceRoleCompression(
            in_features=self.freq_basis_dim,
            role_dim=self.config.role_dim,
            nonneg=self.config.role_nonneg,
            input_norm=self.config.role_input_norm,
            output_norm=self.config.role_output_norm,
            init_mode=self.config.role_init_mode,
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
        self.hierarchical_evidence_integration = HierarchicalEvidenceIntegration(
            role_dim=self.config.role_dim,
            concept_norm=self.config.concept_norm,
        )
        self.time_role_refiner = nn.Sequential(
            nn.LayerNorm(self.config.role_dim),
            nn.Linear(self.config.role_dim, self.config.concept_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.config.concept_dim, self.config.role_dim, bias=False),
        )
        self.freq_role_refiner = nn.Sequential(
            nn.LayerNorm(self.config.role_dim),
            nn.Linear(self.config.role_dim, self.config.concept_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.config.concept_dim, self.config.role_dim, bias=False),
        )
        self.global_semantic_proj = nn.Linear(self.config.concept_dim, self.config.concept_dim, bias=False)
        self.global_semantic_out = nn.Linear(self.config.concept_dim, self.config.concept_dim, bias=False)
        self.global_semantic_norm = nn.LayerNorm(self.config.concept_dim)
        self.evidence_path_gate_weight: Optional[nn.Parameter] = None
        if str(getattr(self.config, "prototype_concept_input", "h")) == "dual_relative_learned":
            # Zero-init keeps the initial decision as an unbiased 50/50 evidence
            # average without consuming RNG; CE then learns which interpretable
            # evidence-quality coordinates are reliable.
            self.evidence_path_gate_weight = nn.Parameter(torch.zeros(4, 2))
        self.local_anomaly_residual_mode = str(
            getattr(self.config, "local_anomaly_residual_mode", "direct")
        )
        self.anomaly_residual_norm = nn.LayerNorm(self.config.concept_dim)
        if self.local_anomaly_residual_mode in {"mlp", "direct_delta"}:
            self.anomaly_residual_proj = nn.Linear(
                self.config.concept_dim,
                self.config.concept_dim,
                bias=False,
            )
        else:
            self.anomaly_residual_proj = None
        self.semantic_residual_init = str(
            getattr(self.config, "semantic_residual_init", "anomaly_zero")
        )
        if self.semantic_residual_init == "zero":
            nn.init.zeros_(self.time_role_refiner[-1].weight)
            nn.init.zeros_(self.freq_role_refiner[-1].weight)
            nn.init.zeros_(self.global_semantic_out.weight)
        elif self.semantic_residual_init == "small":
            nn.init.xavier_uniform_(self.time_role_refiner[-1].weight, gain=0.1)
            nn.init.xavier_uniform_(self.freq_role_refiner[-1].weight, gain=0.1)
            nn.init.xavier_uniform_(self.global_semantic_out.weight, gain=0.1)
            if self.anomaly_residual_proj is not None:
                nn.init.xavier_uniform_(self.anomaly_residual_proj.weight, gain=0.1)
        if self.semantic_residual_init in {"zero", "anomaly_zero"}:
            if self.anomaly_residual_proj is not None:
                nn.init.zeros_(self.anomaly_residual_proj.weight)
        self.global_deep_enabled = bool(
            self.config.transparent_backbone_enabled and self.config.transparent_backbone_weight > 0.0
        )
        self.transparent_backbone = None
        self.transparent_time_backbone = None
        self.transparent_freq_backbone = None
        self.transparent_time_basis_backbone = None
        self.transparent_feature_dim = 0
        self.transparent_time_feature_dim = 0
        self.transparent_freq_feature_dim = 0
        self.transparent_feature_names: list[str] = []
        self.transparent_time_feature_names: list[str] = []
        self.transparent_freq_feature_names: list[str] = []
        self.transparent_time_feature_norm: Optional[nn.LayerNorm] = None
        self.transparent_time_feature_proj: Optional[nn.Linear] = None
        self.transparent_freq_feature_norm: Optional[nn.LayerNorm] = None
        self.transparent_freq_feature_proj: Optional[nn.Linear] = None
        self.transparent_time_to_role: Optional[nn.Linear] = None
        self.transparent_freq_to_role: Optional[nn.Linear] = None
        self.transparent_time_basis_feature_norm: Optional[nn.LayerNorm] = None
        self.transparent_time_basis_feature_proj: Optional[nn.Linear] = None
        self.transparent_time_basis_to_role: Optional[nn.Linear] = None
        self.transparent_time_basis_mix_logit: Optional[nn.Parameter] = None
        self.residual_enabled = float(self.config.classifier.residual_weight) > 0.0
        self.residual_input_mode = str(self.config.classifier.residual_input)
        self.raw_feature_residual_topk: Optional[int] = None
        self.raw_feature_residual_rank: Optional[int] = None
        self.raw_feature_residual_lowrank_proj: Optional[nn.Linear] = None
        self.raw_feature_residual_activation: Optional[nn.Module] = None
        self.semantic_coord_token_dim: Optional[int] = None
        self.semantic_coord_slot_count: int = 0
        self.semantic_coord_token_norm: Optional[nn.LayerNorm] = None
        self.semantic_coord_token_proj: Optional[nn.Linear] = None
        self.semantic_coord_token_refine: Optional[nn.Linear] = None
        self.semantic_coord_slot_fuse: Optional[nn.Linear] = None
        self.semantic_coord_slot_norm: Optional[nn.LayerNorm] = None
        self.semantic_coord_residual_proj: Optional[nn.Linear] = None
        self.semantic_tokens_attn_dim: Optional[int] = None
        self.semantic_tokens_attn_norm: Optional[nn.LayerNorm] = None
        self.semantic_tokens_attn_proj: Optional[nn.Linear] = None
        self.semantic_tokens_self_attn: Optional[nn.MultiheadAttention] = None
        self.semantic_tokens_self_attn_norm: Optional[nn.LayerNorm] = None
        self.semantic_tokens_ffn_in: Optional[nn.Linear] = None
        self.semantic_tokens_ffn_out: Optional[nn.Linear] = None
        self.semantic_tokens_ffn_norm: Optional[nn.LayerNorm] = None
        self.semantic_tokens_query_proj: Optional[nn.Linear] = None
        self.semantic_tokens_key_proj: Optional[nn.Linear] = None
        self.semantic_tokens_value_proj: Optional[nn.Linear] = None
        self.semantic_tokens_out_norm: Optional[nn.LayerNorm] = None
        self.semantic_tokens_residual_proj: Optional[nn.Linear] = None
        residual_rng_state = None
        residual_cuda_rng_state = None
        residual_init_mode_for_rng = str(getattr(self.config.classifier, "residual_init", "default"))
        if self.residual_enabled and residual_init_mode_for_rng in {"zero", "dct", "feature_hash"}:
            # Deterministic residual branches should not perturb downstream
            # initialization.  nn.Linear consumes RNG during construction even
            # when its weights are overwritten immediately afterwards.
            residual_rng_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                residual_cuda_rng_state = torch.cuda.get_rng_state_all()
        if self.residual_enabled:
            residual_proj_in_dim = int(self.feature_dim)
            if self.residual_input_mode == "fixed_semantic_coord":
                token_dim = self.config.classifier.residual_rank
                if token_dim is None:
                    token_dim = int(self.config.concept_dim)
                token_dim = max(16, int(token_dim))
                self.semantic_coord_token_dim = token_dim
                self.semantic_coord_token_norm = nn.LayerNorm(self.semantic_coord_stats_dim)
                self.semantic_coord_token_proj = nn.Linear(
                    self.semantic_coord_stats_dim,
                    token_dim,
                    bias=False,
                )
                self.semantic_coord_token_refine = nn.Linear(
                    token_dim,
                    token_dim,
                    bias=False,
                )
                self.semantic_coord_slot_count = 3
                self.semantic_coord_slot_fuse = nn.Linear(
                    token_dim * self.semantic_coord_slot_count,
                    token_dim,
                    bias=False,
                )
                self.semantic_coord_slot_norm = nn.LayerNorm(token_dim)
                self.semantic_coord_residual_proj = nn.Linear(
                    token_dim,
                    self.config.concept_dim,
                    bias=False,
                )
                self.raw_feature_residual_norm = None
                self.raw_feature_residual_proj = None
            elif self.residual_input_mode in {"semantic_tokens_attn", "semantic_tokens_query"}:
                token_dim = self.config.classifier.residual_rank
                if token_dim is None:
                    token_dim = int(self.config.concept_dim)
                token_dim = max(16, int(token_dim))
                num_heads = 4 if token_dim % 4 == 0 else 1
                self.semantic_tokens_attn_dim = token_dim
                self.semantic_tokens_attn_norm = nn.LayerNorm(self.semantic_coord_stats_dim)
                self.semantic_tokens_attn_proj = nn.Linear(
                    self.semantic_coord_stats_dim,
                    token_dim,
                    bias=False,
                )
                if self.residual_input_mode == "semantic_tokens_attn":
                    self.semantic_tokens_self_attn = nn.MultiheadAttention(
                        embed_dim=token_dim,
                        num_heads=num_heads,
                        batch_first=True,
                    )
                    self.semantic_tokens_self_attn_norm = nn.LayerNorm(token_dim)
                    self.semantic_tokens_ffn_in = nn.Linear(token_dim, 2 * token_dim, bias=False)
                    self.semantic_tokens_ffn_out = nn.Linear(2 * token_dim, token_dim, bias=False)
                    self.semantic_tokens_ffn_norm = nn.LayerNorm(token_dim)
                self.semantic_tokens_query_proj = nn.Linear(
                    self.config.concept_dim,
                    token_dim,
                    bias=False,
                )
                self.semantic_tokens_key_proj = nn.Linear(token_dim, token_dim, bias=False)
                self.semantic_tokens_value_proj = nn.Linear(token_dim, token_dim, bias=False)
                self.semantic_tokens_out_norm = nn.LayerNorm(token_dim)
                self.semantic_tokens_residual_proj = nn.Linear(
                    token_dim,
                    self.config.concept_dim,
                    bias=False,
                )
                self.raw_feature_residual_norm = None
                self.raw_feature_residual_proj = None
            elif self.residual_input_mode == "global_feature":
                residual_proj_in_dim = int(self.global_feature_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "global_feature_mlp":
                residual_proj_in_dim = int(self.global_feature_dim)
                global_hidden_dim = max(2 * int(self.config.concept_dim), 64)
                self.raw_feature_residual_rank = int(global_hidden_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_lowrank_proj = nn.Linear(
                    residual_proj_in_dim,
                    global_hidden_dim,
                    bias=False,
                )
                self.raw_feature_residual_activation = nn.GELU()
                self.raw_feature_residual_proj = nn.Linear(
                    global_hidden_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "semantic_stats":
                residual_proj_in_dim = int(self.semantic_stats_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "semantic_stats_mlp":
                residual_proj_in_dim = int(self.semantic_stats_dim)
                semantic_hidden_dim = max(4 * int(self.config.concept_dim), 128)
                self.raw_feature_residual_rank = int(semantic_hidden_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_lowrank_proj = nn.Linear(
                    residual_proj_in_dim,
                    semantic_hidden_dim,
                    bias=False,
                )
                self.raw_feature_residual_activation = nn.GELU()
                self.raw_feature_residual_proj = nn.Linear(
                    semantic_hidden_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "local_anomaly_summary":
                residual_proj_in_dim = int(self.local_anomaly_summary_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "local_anomaly_profile":
                residual_proj_in_dim = int(self.local_anomaly_profile_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim,
                    self.config.concept_dim,
                    bias=False,
                )
            elif self.residual_input_mode == "raw_feature_topk":
                topk = self.config.classifier.residual_topk
                if topk is None:
                    topk = self.config.top_k_features if self.config.top_k_features is not None else 512
                self.raw_feature_residual_topk = min(max(1, int(topk)), self.feature_dim)
                residual_proj_in_dim = int(self.raw_feature_residual_topk)
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim, self.config.concept_dim, bias=False
                )
            elif self.residual_input_mode == "raw_feature_lowrank":
                rank = self.config.classifier.residual_rank
                if rank is None:
                    rank = 48
                self.raw_feature_residual_rank = min(max(1, int(rank)), self.feature_dim)
                self.raw_feature_residual_norm = nn.LayerNorm(self.feature_dim)
                self.raw_feature_residual_lowrank_proj = nn.Linear(
                    self.feature_dim,
                    self.raw_feature_residual_rank,
                    bias=False,
                )
                self.raw_feature_residual_proj = nn.Linear(
                    self.raw_feature_residual_rank,
                    self.config.concept_dim,
                    bias=False,
                )
            else:
                self.raw_feature_residual_norm = nn.LayerNorm(residual_proj_in_dim)
                self.raw_feature_residual_proj = nn.Linear(
                    residual_proj_in_dim, self.config.concept_dim, bias=False
                )
            self._init_residual_projection_for_stability()
            if residual_rng_state is not None:
                torch.random.set_rng_state(residual_rng_state)
                if residual_cuda_rng_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(residual_cuda_rng_state)
        else:
            self.raw_feature_residual_norm = None
            self.raw_feature_residual_proj = None
            self.raw_feature_residual_lowrank_proj = None
            self.raw_feature_residual_activation = None
            self.semantic_coord_token_dim = None
            self.semantic_coord_slot_count = 0
            self.semantic_coord_token_norm = None
            self.semantic_coord_token_proj = None
            self.semantic_coord_token_refine = None
            self.semantic_coord_slot_fuse = None
            self.semantic_coord_slot_norm = None
            self.semantic_coord_residual_proj = None
            self.semantic_tokens_attn_dim = None
            self.semantic_tokens_attn_norm = None
            self.semantic_tokens_attn_proj = None
            self.semantic_tokens_self_attn = None
            self.semantic_tokens_self_attn_norm = None
            self.semantic_tokens_ffn_in = None
            self.semantic_tokens_ffn_out = None
            self.semantic_tokens_ffn_norm = None
            self.semantic_tokens_query_proj = None
            self.semantic_tokens_key_proj = None
            self.semantic_tokens_value_proj = None
            self.semantic_tokens_out_norm = None
            self.semantic_tokens_residual_proj = None
        residual_summary: Dict[str, Any] = {
            "enabled": bool(self.residual_enabled),
            "input_mode": str(self.residual_input_mode) if self.residual_enabled else "off",
            "weight": float(self.config.classifier.residual_weight),
            "max_ratio": float(self.config.classifier.residual_max_ratio),
            "alignment_mode": str(getattr(self.config.classifier, "residual_alignment_mode", "off")),
        }
        if self.residual_enabled and self.residual_input_mode == "fixed_semantic_coord":
            residual_summary["semantic_coord"] = {
                "token_count": int(self.semantic_coord_token_count),
                "stats": ["mean", "std", "maxabs", "global"],
                "token_dim": int(self.semantic_coord_token_dim or 0),
                "pooling": [
                    "uniform_mean",
                    "anomaly_weighted(std+maxabs)",
                    "drift_weighted(|mean-global|)",
                ],
                "slot_count": int(self.semantic_coord_slot_count),
            }
        if self.residual_enabled and self.residual_input_mode in {
            "semantic_tokens_attn",
            "semantic_tokens_query",
        }:
            residual_summary[self.residual_input_mode] = {
                "token_count": int(self.semantic_coord_token_count),
                "token_dim": int(self.semantic_tokens_attn_dim or 0),
                "stats": ["mean", "std", "maxabs", "global"],
                "pooling": (
                    "self_attention_then_core_query"
                    if self.residual_input_mode == "semantic_tokens_attn"
                    else "core_query_attention"
                ),
            }
        self.active_component_summary["residual_branch"] = residual_summary
        self.time_prototype_head: Optional[PrototypeHead] = None
        self.freq_prototype_head: Optional[PrototypeHead] = None
        self.joint_prototype_head = PrototypeHead(
            num_classes=self.config.num_classes,
            concept_dim=self.config.concept_dim,
            num_prototypes_per_class=self.config.num_prototypes_per_class,
            temperature=self.config.prototype_temperature,
            class_pool_mode=self.config.prototype_class_pool_mode,
            class_anchor_mode=self.config.prototype_class_anchor_mode,
            residual_score_mode=self.config.prototype_residual_score_mode,
            residual_logit_mode=self.config.prototype_residual_logit_mode,
            residual_logit_weight=self.config.prototype_residual_logit_weight,
            adaptive_class_temperature_enabled=self.config.adaptive_class_temperature_enabled,
            adaptive_class_temperature_target_neff=self.config.adaptive_class_temperature_target_neff,
            adaptive_class_temperature_min_scale=self.config.adaptive_class_temperature_min_scale,
            adaptive_class_temperature_max_scale=self.config.adaptive_class_temperature_max_scale,
            adaptive_effective_k_enabled=self.config.adaptive_effective_k_enabled,
            adaptive_effective_k_ready_count=self.config.adaptive_effective_k_ready_count,
            adaptive_effective_k_min=self.config.adaptive_effective_k_min,
            adaptive_effective_k_penalty=self.config.adaptive_effective_k_penalty,
            assignment_temperature=self.config.prototype_assignment_temperature,
            assignment_mode=self.config.prototype.assignment_mode,
            balance_weight=self.config.prototype.balance_weight,
            logit_scale_init=self.config.prototype_logit_scale_init,
            init_mode=self.config.prototype_init_mode,
            init_scale=self.config.prototype_init_scale,
        )
        # Keep cooperative branches truly inactive when cooperative mode is disabled:
        # do not instantiate their parameters to avoid initialization drift.
        if self.cooperative_enabled:
            self.time_prototype_head = PrototypeHead(
                num_classes=self.config.num_classes,
                concept_dim=self.config.role_dim,
                num_prototypes_per_class=self.config.num_prototypes_per_class,
                temperature=self.config.prototype_temperature,
                class_pool_mode=self.config.prototype_class_pool_mode,
                class_anchor_mode=self.config.prototype_class_anchor_mode,
                residual_score_mode=self.config.prototype_residual_score_mode,
                residual_logit_mode=self.config.prototype_residual_logit_mode,
                residual_logit_weight=self.config.prototype_residual_logit_weight,
                adaptive_class_temperature_enabled=self.config.adaptive_class_temperature_enabled,
                adaptive_class_temperature_target_neff=self.config.adaptive_class_temperature_target_neff,
                adaptive_class_temperature_min_scale=self.config.adaptive_class_temperature_min_scale,
                adaptive_class_temperature_max_scale=self.config.adaptive_class_temperature_max_scale,
                adaptive_effective_k_enabled=self.config.adaptive_effective_k_enabled,
                adaptive_effective_k_ready_count=self.config.adaptive_effective_k_ready_count,
                adaptive_effective_k_min=self.config.adaptive_effective_k_min,
                adaptive_effective_k_penalty=self.config.adaptive_effective_k_penalty,
                assignment_temperature=self.config.prototype_assignment_temperature,
                assignment_mode=self.config.prototype.assignment_mode,
                balance_weight=self.config.prototype.balance_weight,
                logit_scale_init=self.config.prototype_logit_scale_init,
                init_mode=self.config.prototype_init_mode,
                init_scale=self.config.prototype_init_scale,
            )
            self.freq_prototype_head = PrototypeHead(
                num_classes=self.config.num_classes,
                concept_dim=self.config.role_dim,
                num_prototypes_per_class=self.config.num_prototypes_per_class,
                temperature=self.config.prototype_temperature,
                class_pool_mode=self.config.prototype_class_pool_mode,
                class_anchor_mode=self.config.prototype_class_anchor_mode,
                residual_score_mode=self.config.prototype_residual_score_mode,
                residual_logit_mode=self.config.prototype_residual_logit_mode,
                residual_logit_weight=self.config.prototype_residual_logit_weight,
                adaptive_class_temperature_enabled=self.config.adaptive_class_temperature_enabled,
                adaptive_class_temperature_target_neff=self.config.adaptive_class_temperature_target_neff,
                adaptive_class_temperature_min_scale=self.config.adaptive_class_temperature_min_scale,
                adaptive_class_temperature_max_scale=self.config.adaptive_class_temperature_max_scale,
                adaptive_effective_k_enabled=self.config.adaptive_effective_k_enabled,
                adaptive_effective_k_ready_count=self.config.adaptive_effective_k_ready_count,
                adaptive_effective_k_min=self.config.adaptive_effective_k_min,
                adaptive_effective_k_penalty=self.config.adaptive_effective_k_penalty,
                assignment_temperature=self.config.prototype_assignment_temperature,
                assignment_mode=self.config.prototype.assignment_mode,
                balance_weight=self.config.prototype.balance_weight,
                logit_scale_init=self.config.prototype_logit_scale_init,
                init_mode=self.config.prototype_init_mode,
                init_scale=self.config.prototype_init_scale,
            )
        self.cooperative_prototype_fusion: Optional[CooperativePrototypeFusion]
        if self.cooperative_enabled:
            self.cooperative_prototype_fusion = CooperativePrototypeFusion(
                weight_floor=float(self.config.cooperative_weight_floor),
                max_share=float(self.config.cooperative_max_share),
                evidence_mix=float(self.config.cooperative_evidence_mix),
                confidence_mix=float(self.config.cooperative_confidence_mix),
                agreement_mix=float(self.config.cooperative_agreement_mix),
                logit_scale_min=float(self.config.cooperative_logit_scale_min),
            )
        else:
            # Deprecated compatibility path: keep joint-only route when cooperative fusion is disabled.
            self.cooperative_prototype_fusion = None
        self.prototype_head = self.joint_prototype_head
        # Compatibility alias for legacy tests and callback assumptions.
        self.prototype_bank = self.prototype_head
        self.active_component_summary["prototype_init"] = {
            "mode": str(self.config.prototype_init_mode),
            "scale": float(self.config.prototype_init_scale),
        }

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
        if self.config.transparent_backbone_enabled:
            transparent_time_args = self._build_transparent_backbone_args(branch="time")
            transparent_freq_args = self._build_transparent_backbone_args(branch="freq")
            self.transparent_time_backbone = TransparentGlobalBackbone(transparent_time_args)
            self.transparent_freq_backbone = TransparentGlobalBackbone(transparent_freq_args)
            # Legacy compatibility alias.
            self.transparent_backbone = self.transparent_time_backbone

            self.transparent_time_feature_dim = int(self.transparent_time_backbone.channel_for_classifier)
            self.transparent_freq_feature_dim = int(self.transparent_freq_backbone.channel_for_classifier)
            self.transparent_feature_dim = int(
                self.transparent_time_feature_dim + self.transparent_freq_feature_dim
            )
            self.transparent_time_feature_norm = nn.LayerNorm(self.transparent_time_feature_dim)
            self.transparent_time_feature_proj = nn.Linear(
                self.transparent_time_feature_dim,
                self.config.concept_dim,
                bias=False,
            )
            self.transparent_freq_feature_norm = nn.LayerNorm(self.transparent_freq_feature_dim)
            self.transparent_freq_feature_proj = nn.Linear(
                self.transparent_freq_feature_dim,
                self.config.concept_dim,
                bias=False,
            )
            self.transparent_time_to_role = nn.Linear(self.config.concept_dim, self.config.role_dim, bias=False)
            self.transparent_freq_to_role = nn.Linear(self.config.concept_dim, self.config.role_dim, bias=False)
            projection_init_mode = str(self.config.transparent_projection_init_mode)
            if projection_init_mode != "random_normal":
                self._init_transparent_projection_with_dct(projection_init_mode)
            if bool(getattr(self.config, "transparent_time_dual_basis_enabled", False)):
                basis_args = self._build_transparent_backbone_args(branch="time")
                basis_args.deterministic_init = True
                self.transparent_time_basis_backbone = TransparentGlobalBackbone(basis_args)
                self.transparent_time_basis_feature_norm = nn.LayerNorm(self.transparent_time_feature_dim)
                self.transparent_time_basis_feature_proj = nn.Linear(
                    self.transparent_time_feature_dim,
                    self.config.concept_dim,
                    bias=False,
                )
                self.transparent_time_basis_to_role = nn.Linear(
                    self.config.concept_dim,
                    self.config.role_dim,
                    bias=False,
                )
                self._init_linear_from_dct(self.transparent_time_basis_feature_proj)
                self._init_linear_from_dct(self.transparent_time_basis_to_role)
                self.transparent_time_basis_mix_logit = nn.Parameter(torch.zeros(()))
            self.transparent_time_feature_names = self._build_transparent_feature_names(
                feature_names=self.config.transparent_backbone_features,
                channels_per_feature=int(self.transparent_time_backbone.channel_for_feature),
            )
            self.transparent_freq_feature_names = self._build_transparent_feature_names(
                feature_names=self.config.transparent_backbone_freq_features,
                channels_per_feature=int(self.transparent_freq_backbone.channel_for_feature),
            )
            self.transparent_feature_names = (
                [f"time.{name}" for name in self.transparent_time_feature_names]
                + [f"freq.{name}" for name in self.transparent_freq_feature_names]
            )
            self.active_component_summary["transparent_backbone"] = {
                "enabled": True,
                "mode": "dual_deep_global_tf",
                "branches": ["time_global_deep", "freq_global_deep"],
                "time": {
                    "layers": int(self.config.transparent_backbone_layers),
                    "modules": list(self.config.transparent_backbone_modules),
                    "features": list(self.config.transparent_backbone_features),
                    "deterministic_init": bool(
                        self.config.transparent_backbone_time_deterministic_init
                        if self.config.transparent_backbone_time_deterministic_init is not None
                        else self.config.transparent_backbone_deterministic_init
                    ),
                    "feature_norm": str(self.config.transparent_backbone_feature_norm),
                    "projection_init": str(self.config.transparent_projection_init_mode),
                    "dual_basis_enabled": bool(
                        getattr(self.config, "transparent_time_dual_basis_enabled", False)
                    ),
                    "feature_dim": int(self.transparent_time_feature_dim),
                },
                "freq": {
                    "layers": int(self.config.transparent_backbone_layers),
                    "modules": list(self.config.transparent_backbone_freq_modules),
                    "features": list(self.config.transparent_backbone_freq_features),
                    "deterministic_init": bool(
                        self.config.transparent_backbone_freq_deterministic_init
                        if self.config.transparent_backbone_freq_deterministic_init is not None
                        else self.config.transparent_backbone_deterministic_init
                    ),
                    "feature_norm": str(self.config.transparent_backbone_feature_norm),
                    "projection_init": str(self.config.transparent_projection_init_mode),
                    "feature_dim": int(self.transparent_freq_feature_dim),
                },
                "feature_dim": int(self.transparent_feature_dim),
            }
        else:
            self.active_component_summary["transparent_backbone"] = {"enabled": False}

    @staticmethod
    def _init_linear_from_dct(linear: nn.Linear) -> None:
        basis = EvidenceRoleCompression._build_dct_basis(
            in_features=int(linear.in_features),
            role_dim=int(linear.out_features),
        ).transpose(0, 1)
        with torch.no_grad():
            linear.weight.copy_(basis.to(device=linear.weight.device, dtype=linear.weight.dtype))

    def _init_transparent_projection_with_dct(self, mode: str) -> None:
        if mode == "dct":
            modules = [
                self.transparent_time_feature_proj,
                self.transparent_freq_feature_proj,
                self.transparent_time_to_role,
                self.transparent_freq_to_role,
            ]
        elif mode == "dct_feature":
            modules = [
                self.transparent_time_feature_proj,
                self.transparent_freq_feature_proj,
            ]
        elif mode == "dct_to_role":
            modules = [
                self.transparent_time_to_role,
                self.transparent_freq_to_role,
            ]
        else:
            raise ValueError(f"Unsupported transparent_projection_init_mode={mode!r}.")
        for module in modules:
            if module is not None:
                self._init_linear_from_dct(module)

    def _feature_mask_on(self, device: torch.device) -> torch.Tensor:
        return self.feature_mask.to(device=device)

    @staticmethod
    def _summary_topk_count(region_count: int) -> int:
        return max(1, int(round(math.sqrt(float(max(region_count, 1))))))

    @staticmethod
    def _topk_abs_mean(features: torch.Tensor, dim: int, k: int) -> torch.Tensor:
        region_dim = int(features.shape[dim])
        k = min(max(1, int(k)), region_dim)
        topk_values = features.abs().topk(k=k, dim=dim).values
        return topk_values.mean(dim=dim)

    @staticmethod
    def _topk_abs_values(features: torch.Tensor, dim: int, k: int) -> torch.Tensor:
        region_dim = int(features.shape[dim])
        k = min(max(1, int(k)), region_dim)
        return features.abs().topk(k=k, dim=dim).values

    def _sample_virtual_raw_weight(
        self,
        out_dim: int,
        in_dim: int,
        gain: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        limit = math.sqrt(6.0 / float(max(out_dim + in_dim, 1))) * float(gain)
        generator = None
        if self.config.classifier.residual_init_seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.config.classifier.residual_init_seed))
        weight = torch.empty(int(out_dim), int(in_dim), dtype=dtype, device="cpu")
        weight.uniform_(-limit, limit, generator=generator)
        return weight

    @staticmethod
    def _feature_hash_init_weight(
        out_dim: int,
        in_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Deterministic, seed-independent initialization for semantic residual branches.
        weight = torch.zeros(int(out_dim), int(in_dim), dtype=dtype, device="cpu")
        fan_in = torch.zeros(int(out_dim), dtype=torch.float32, device="cpu")
        out_dim_i = int(out_dim)
        for j in range(int(in_dim)):
            h1 = ((j * 1315423911) + 2654435761) % out_dim_i
            s1 = 1.0 if ((j * 2246822519 + 3266489917) & 1) == 0 else -1.0
            weight[h1, j] += s1
            fan_in[h1] += 1.0
            h2 = ((j * 374761393) + 668265263) % out_dim_i
            if h2 != h1:
                s2 = 1.0 if ((j * 1597334677 + 3812015801) & 1) == 0 else -1.0
                weight[h2, j] += s2
                fan_in[h2] += 1.0
        scale = fan_in.clamp_min(1.0).sqrt().unsqueeze(1).to(weight.dtype)
        weight = weight / scale
        return weight

    def _projected_raw_global_init_weight(self, weight: torch.Tensor) -> torch.Tensor:
        concept_dim = int(self.config.concept_dim)
        raw_dim = int(self.feature_dim)
        virtual_raw = self._sample_virtual_raw_weight(
            out_dim=concept_dim,
            in_dim=raw_dim,
            gain=0.5,
            dtype=weight.dtype,
        )

        channels = int(self.config.in_channels)
        time_ops = len(self.time_operator_bank.operator_names)
        time_indicators = len(self.config.time_indicators)
        time_patches = int(self.config.time_patch_count)
        freq_ops = len(self.freq_operator_bank.operator_names)
        freq_indicators = len(self.config.freq_indicators)
        freq_bands = int(self.config.freq_band_count)
        time_count = channels * time_ops * time_patches * time_indicators

        time_part = virtual_raw[:, :time_count].view(
            concept_dim,
            channels,
            time_ops,
            time_patches,
            time_indicators,
        )
        freq_part = virtual_raw[:, time_count:].view(
            concept_dim,
            channels,
            freq_ops,
            freq_bands,
            freq_indicators,
        )
        time_global = time_part.mean(dim=3) * math.sqrt(float(max(time_patches, 1)))
        freq_global = freq_part.mean(dim=3) * math.sqrt(float(max(freq_bands, 1)))
        projected = torch.cat(
            [time_global.reshape(concept_dim, -1), freq_global.reshape(concept_dim, -1)],
            dim=1,
        )
        return projected.to(device=weight.device, dtype=weight.dtype)

    def _init_residual_projection_for_stability(self) -> None:
        if not self.residual_enabled:
            return
        residual_init_mode = str(self.config.classifier.residual_init)
        # Legacy-compatible behavior:
        # For simple one-layer residual branches, keep PyTorch Linear default init.
        # Historical high-performing runs follow this path.
        if (
            residual_init_mode == "default"
            and self.raw_feature_residual_lowrank_proj is None
            and self.residual_input_mode != "fixed_semantic_coord"
        ):
            return
        if self.residual_input_mode == "fixed_semantic_coord":
            if self.semantic_coord_token_proj is not None:
                nn.init.xavier_uniform_(self.semantic_coord_token_proj.weight, gain=1.0)
            if self.semantic_coord_token_refine is not None:
                nn.init.xavier_uniform_(self.semantic_coord_token_refine.weight, gain=0.8)
            if self.semantic_coord_slot_fuse is not None:
                nn.init.xavier_uniform_(self.semantic_coord_slot_fuse.weight, gain=0.6)
            if self.semantic_coord_residual_proj is not None:
                nn.init.xavier_uniform_(self.semantic_coord_residual_proj.weight, gain=0.1)
            return
        if self.raw_feature_residual_lowrank_proj is not None:
            nn.init.xavier_uniform_(self.raw_feature_residual_lowrank_proj.weight, gain=1.0)
        if self.raw_feature_residual_proj is None:
            return
        if (
            self.residual_input_mode == "global_feature"
            and residual_init_mode == "projected_raw"
            and self.raw_feature_residual_lowrank_proj is None
        ):
            with torch.no_grad():
                projected = self._projected_raw_global_init_weight(self.raw_feature_residual_proj.weight)
                self.raw_feature_residual_proj.weight.copy_(projected)
        elif residual_init_mode == "zero" and self.raw_feature_residual_lowrank_proj is None:
            with torch.no_grad():
                self.raw_feature_residual_proj.weight.zero_()
        elif residual_init_mode == "dct" and self.raw_feature_residual_lowrank_proj is None:
            self._init_linear_from_dct(self.raw_feature_residual_proj)
        elif residual_init_mode == "feature_hash" and self.raw_feature_residual_lowrank_proj is None:
            with torch.no_grad():
                hashed = self._feature_hash_init_weight(
                    out_dim=int(self.raw_feature_residual_proj.weight.shape[0]),
                    in_dim=int(self.raw_feature_residual_proj.weight.shape[1]),
                    dtype=self.raw_feature_residual_proj.weight.dtype,
                )
                self.raw_feature_residual_proj.weight.copy_(
                    hashed.to(
                        device=self.raw_feature_residual_proj.weight.device,
                        dtype=self.raw_feature_residual_proj.weight.dtype,
                    )
                )
        elif (
            self.residual_input_mode == "raw_feature_lowrank"
            and residual_init_mode == "raw_svd"
            and self.raw_feature_residual_lowrank_proj is not None
        ):
            rank = int(self.raw_feature_residual_lowrank_proj.weight.shape[0])
            virtual_raw = self._sample_virtual_raw_weight(
                out_dim=int(self.config.concept_dim),
                in_dim=int(self.feature_dim),
                gain=0.5,
                dtype=self.raw_feature_residual_proj.weight.dtype,
            )
            u, s, vh = torch.linalg.svd(virtual_raw, full_matrices=False)
            effective_rank = min(rank, int(s.shape[0]))
            lowrank_weight = torch.zeros_like(self.raw_feature_residual_lowrank_proj.weight)
            out_weight = torch.zeros_like(self.raw_feature_residual_proj.weight)
            lowrank_weight[:effective_rank] = vh[:effective_rank]
            out_weight[:, :effective_rank] = u[:, :effective_rank] * s[:effective_rank].unsqueeze(0)
            with torch.no_grad():
                self.raw_feature_residual_lowrank_proj.weight.copy_(
                    lowrank_weight.to(
                        device=self.raw_feature_residual_lowrank_proj.weight.device,
                        dtype=self.raw_feature_residual_lowrank_proj.weight.dtype,
                    )
                )
                self.raw_feature_residual_proj.weight.copy_(
                    out_weight.to(
                        device=self.raw_feature_residual_proj.weight.device,
                        dtype=self.raw_feature_residual_proj.weight.dtype,
                    )
                )
        else:
            nn.init.xavier_uniform_(self.raw_feature_residual_proj.weight, gain=0.5)


    def _should_use_eval_mc(self) -> bool:
        return (
            (not self.training)
            and int(self.config.eval_mc_samples) > 1
            and (
                str(self.config.time_patch_mode) in {"random", "random_stratified"}
                or str(self.config.freq_band_mode) in {"random", "random_stratified"}
            )
        )

    def _should_use_region_ensemble(self) -> bool:
        return int(self.config.region_ensemble_samples) > 1

    def _build_transparent_backbone_args(self, branch: str = "time") -> SimpleNamespace:
        if str(branch) == "freq":
            module_names = list(self.config.transparent_backbone_freq_modules)
            feature_names = list(self.config.transparent_backbone_freq_features)
        else:
            module_names = list(self.config.transparent_backbone_modules)
            feature_names = list(self.config.transparent_backbone_features)
        feature_norm_mode = str(self.config.transparent_backbone_feature_norm)
        feature_norm_eps = float(self.config.transparent_backbone_feature_norm_eps)
        # Keep backward-compatible epsilon for the legacy running-stat normalizer.
        if feature_norm_mode == "legacy_running" and feature_norm_eps < 1e-3:
            feature_norm_eps = 0.1
        signal_processing_configs = {
            f"layer{layer_idx + 1}": module_names
            for layer_idx in range(int(self.config.transparent_backbone_layers))
        }
        if str(branch) == "freq":
            deterministic_init = self.config.transparent_backbone_freq_deterministic_init
        else:
            deterministic_init = self.config.transparent_backbone_time_deterministic_init
        if deterministic_init is None:
            deterministic_init = bool(self.config.transparent_backbone_deterministic_init)
        return SimpleNamespace(
            name="TransparentGlobalBackbone",
            type="X_model",
            device=str(self.config.device),
            in_dim=int(self.config.input_length),
            out_dim=int(self.config.input_length),
            in_channels=int(self.config.in_channels),
            out_channels=int(self.config.transparent_backbone_out_channels),
            scale=int(self.config.transparent_backbone_scale),
            skip_connection=bool(self.config.transparent_backbone_skip_connection),
            deterministic_init=bool(deterministic_init),
            feature_norm_mode=feature_norm_mode,
            feature_norm_eps=feature_norm_eps,
            signal_processing_configs=signal_processing_configs,
            feature_extractor_configs=feature_names,
            num_classes=self.config.num_classes,
            f_c_mu=0.0,
            f_c_sigma=0.1,
            f_b_mu=0.0,
            f_b_sigma=0.1,
        )

    @staticmethod
    def _build_transparent_feature_names(
        feature_names: list[str],
        channels_per_feature: int,
    ) -> list[str]:
        names: list[str] = []
        for feature_name in feature_names:
            for channel_index in range(int(channels_per_feature)):
                names.append(f"tspn.{feature_name}.ch{channel_index}")
        return names

    def _build_semantic_coord_names(self) -> list[str]:
        names: list[str] = []
        for channel_index in range(int(self.config.in_channels)):
            for operator_name in self.time_operator_bank.operator_names:
                for indicator_name in self.config.time_indicators:
                    names.append(f"time.ch{channel_index}.{operator_name}.{indicator_name}")
            for operator_name in self.freq_operator_bank.operator_names:
                for indicator_name in self.config.freq_indicators:
                    names.append(f"freq.ch{channel_index}.{operator_name}.{indicator_name}")
        return names

    def _semantic_coord_residual_forward(
        self,
        semantic_coord_tokens: torch.Tensor,
        _global_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.semantic_coord_token_norm is None
            or self.semantic_coord_token_proj is None
            or self.semantic_coord_token_refine is None
            or self.semantic_coord_slot_fuse is None
            or self.semantic_coord_slot_norm is None
            or self.semantic_coord_residual_proj is None
        ):
            raise RuntimeError("fixed_semantic_coord residual modules are not initialized.")
        token_hidden = self.semantic_coord_token_norm(semantic_coord_tokens)
        token_hidden = self.semantic_coord_token_proj(token_hidden)
        token_hidden = F.gelu(token_hidden)
        token_hidden = token_hidden + self.semantic_coord_token_refine(token_hidden)
        token_hidden = F.layer_norm(token_hidden, (int(token_hidden.shape[-1]),))
        eps = 1e-6
        anomaly_strength = semantic_coord_tokens[..., 1].abs() + semantic_coord_tokens[..., 2].abs()
        drift_strength = (semantic_coord_tokens[..., 0] - semantic_coord_tokens[..., 3]).abs()

        uniform_weights = torch.full_like(anomaly_strength, 1.0 / float(anomaly_strength.shape[1]))
        anomaly_logits = torch.log(anomaly_strength + eps)
        drift_logits = torch.log(drift_strength + eps)
        anomaly_weights = torch.softmax(anomaly_logits, dim=1)
        drift_weights = torch.softmax(drift_logits, dim=1)

        pooled_uniform = torch.sum(uniform_weights.unsqueeze(-1) * token_hidden, dim=1)
        pooled_anomaly = torch.sum(anomaly_weights.unsqueeze(-1) * token_hidden, dim=1)
        pooled_drift = torch.sum(drift_weights.unsqueeze(-1) * token_hidden, dim=1)

        pooled = torch.cat([pooled_uniform, pooled_anomaly, pooled_drift], dim=-1)
        pooled = self.semantic_coord_slot_fuse(pooled)
        pooled = self.semantic_coord_slot_norm(F.gelu(pooled))
        residual = self.semantic_coord_residual_proj(pooled)
        semantic_coord_attention = torch.stack(
            [uniform_weights, anomaly_weights, drift_weights],
            dim=1,
        )
        return torch.nan_to_num(residual), torch.nan_to_num(semantic_coord_attention)

    def _semantic_tokens_attn_forward(
        self,
        semantic_coord_tokens: torch.Tensor,
        h_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.semantic_tokens_attn_norm is None
            or self.semantic_tokens_attn_proj is None
            or self.semantic_tokens_self_attn is None
            or self.semantic_tokens_self_attn_norm is None
            or self.semantic_tokens_ffn_in is None
            or self.semantic_tokens_ffn_out is None
            or self.semantic_tokens_ffn_norm is None
            or self.semantic_tokens_query_proj is None
            or self.semantic_tokens_key_proj is None
            or self.semantic_tokens_value_proj is None
            or self.semantic_tokens_out_norm is None
            or self.semantic_tokens_residual_proj is None
        ):
            raise RuntimeError("semantic_tokens_attn residual modules are not initialized.")
        tokens = self.semantic_tokens_attn_norm(semantic_coord_tokens)
        tokens = F.gelu(self.semantic_tokens_attn_proj(tokens))
        self_attn_out, _ = self.semantic_tokens_self_attn(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )
        tokens = self.semantic_tokens_self_attn_norm(tokens + self_attn_out)
        token_ffn = self.semantic_tokens_ffn_out(F.gelu(self.semantic_tokens_ffn_in(tokens)))
        tokens = self.semantic_tokens_ffn_norm(tokens + token_ffn)

        query = self.semantic_tokens_query_proj(h_core).unsqueeze(1)
        key = self.semantic_tokens_key_proj(tokens)
        value = self.semantic_tokens_value_proj(tokens)
        scale = math.sqrt(float(max(int(key.shape[-1]), 1)))
        attn_logits = torch.sum(query * key, dim=-1) / scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * value, dim=1)
        pooled = self.semantic_tokens_out_norm(pooled)
        residual = self.semantic_tokens_residual_proj(pooled)
        return torch.nan_to_num(residual), torch.nan_to_num(attn_weights)

    def _semantic_tokens_query_forward(
        self,
        semantic_coord_tokens: torch.Tensor,
        h_core: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.semantic_tokens_attn_norm is None
            or self.semantic_tokens_attn_proj is None
            or self.semantic_tokens_query_proj is None
            or self.semantic_tokens_key_proj is None
            or self.semantic_tokens_value_proj is None
            or self.semantic_tokens_out_norm is None
            or self.semantic_tokens_residual_proj is None
        ):
            raise RuntimeError("semantic_tokens_query residual modules are not initialized.")
        tokens = self.semantic_tokens_attn_norm(semantic_coord_tokens)
        tokens = F.gelu(self.semantic_tokens_attn_proj(tokens))
        query = self.semantic_tokens_query_proj(h_core).unsqueeze(1)
        key = self.semantic_tokens_key_proj(tokens)
        value = self.semantic_tokens_value_proj(tokens)
        scale = math.sqrt(float(max(int(key.shape[-1]), 1)))
        attn_logits = torch.sum(query * key, dim=-1) / scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * value, dim=1)
        pooled = self.semantic_tokens_out_norm(pooled)
        residual = self.semantic_tokens_residual_proj(pooled)
        return torch.nan_to_num(residual), torch.nan_to_num(attn_weights)

    @staticmethod
    def _apply_operator_bank_on_regions(
        regions_bcrw: torch.Tensor,
        operator_bank: nn.Module,
    ) -> torch.Tensor:
        if regions_bcrw.ndim != 4:
            raise ValueError(f"Expected [B, C, R, W], got {tuple(regions_bcrw.shape)}.")
        batch_size, num_channels, region_count, region_width = regions_bcrw.shape
        flat = regions_bcrw.permute(0, 2, 1, 3).reshape(batch_size * region_count, num_channels, region_width)
        stacked = operator_bank(flat)
        operator_count = int(stacked.shape[2])
        return (
            stacked.view(batch_size, region_count, num_channels, operator_count, region_width)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )

    def _extract_transparent_backbone_features(
        self,
        backbone: Optional[nn.Module],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if backbone is None:
            raise RuntimeError("Transparent backbone is not enabled.")
        if hasattr(backbone, "extract_features"):
            return backbone.extract_features(x)
        hidden = x
        for layer in backbone.signal_processing_layers:
            hidden = layer(hidden)
        hidden = backbone.feature_extractor_layers(hidden)
        return hidden.view(hidden.shape[0], -1)

    def _build_transparent_backbone_input(
        self,
        x: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        if str(branch) == "time":
            return x
        if str(branch) == "freq":
            x_bcl = to_bcl(x)
            # Frequency branch uses amplitude spectrum semantics; enhancement and
            # indicator extraction are handled by frequency-specific backbone modules/features.
            x_freq = safe_rfft_amplitude(x_bcl)
            return x_freq.permute(0, 2, 1).contiguous()
        raise ValueError(f"Unsupported transparent backbone branch: {branch!r}.")

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

    def _build_region_sampling_keys(
        self,
        x_bcl: torch.Tensor,
        file_ids: torch.Tensor,
    ) -> torch.Tensor:
        file_ids = file_ids.to(device=x_bcl.device, dtype=torch.long).view(-1)
        sample_count = int(x_bcl.shape[0])
        if int(file_ids.numel()) != sample_count:
            if int(file_ids.numel()) == 1:
                file_ids = file_ids.expand(sample_count)
            else:
                file_ids = torch.zeros(sample_count, device=x_bcl.device, dtype=torch.long)
        stride = max(1, int(x_bcl.shape[-1]) // 32)
        coarse = x_bcl[:, :, ::stride]
        mean_q = torch.round(coarse.mean(dim=(1, 2)) * 1e4).to(torch.long)
        std_q = torch.round(coarse.std(dim=(1, 2), unbiased=False) * 1e4).to(torch.long)
        energy_q = torch.round((coarse.square().mean(dim=(1, 2))) * 1e4).to(torch.long)
        keys = file_ids * 1315423911
        keys = keys ^ (mean_q * 2654435761)
        keys = keys ^ (std_q * 2246822519)
        keys = keys ^ (energy_q * 3266489917)
        return keys

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
        h_raw_global_feature: torch.Tensor,
        h_raw_semantic_stats: torch.Tensor,
        h_raw_semantic_coord_tokens: torch.Tensor,
        h_raw_local_anomaly_summary: torch.Tensor,
        h_raw_local_anomaly_profile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if not self.residual_enabled:
            return h_core, h_core.new_zeros(h_core.shape), {
                "topk_indices": None,
                "topk_values": None,
                "semantic_coord_attention": None,
                "residual_ratio_scale": None,
                "residual_alignment_gate": None,
            }
        residual_weight = float(self.config.classifier.residual_weight)
        residual_details: Dict[str, Optional[torch.Tensor]] = {
            "topk_indices": None,
            "topk_values": None,
            "semantic_coord_attention": None,
            "residual_ratio_scale": None,
            "residual_alignment_gate": None,
        }

        residual_input = str(self.config.classifier.residual_input)
        if residual_input == "fixed_semantic_coord":
            residual, semantic_coord_attention = self._semantic_coord_residual_forward(
                h_raw_semantic_coord_tokens,
                h_raw_global_feature,
            )
            residual_details["semantic_coord_attention"] = semantic_coord_attention
        elif residual_input == "semantic_tokens_attn":
            residual, semantic_token_attention = self._semantic_tokens_attn_forward(
                h_raw_semantic_coord_tokens,
                h_core,
            )
            residual_details["semantic_coord_attention"] = semantic_token_attention.unsqueeze(1)
        elif residual_input == "semantic_tokens_query":
            residual, semantic_token_attention = self._semantic_tokens_query_forward(
                h_raw_semantic_coord_tokens,
                h_core,
            )
            residual_details["semantic_coord_attention"] = semantic_token_attention.unsqueeze(1)
        else:
            if self.raw_feature_residual_norm is None or self.raw_feature_residual_proj is None:
                return h_core, h_core.new_zeros(h_core.shape), residual_details
            if residual_input == "raw_feature":
                residual_source = h_raw_full
            elif residual_input in {"global_feature", "global_feature_mlp"}:
                residual_source = h_raw_global_feature
            elif residual_input in {"semantic_stats", "semantic_stats_mlp"}:
                residual_source = h_raw_semantic_stats
            elif residual_input == "local_anomaly_summary":
                residual_source = h_raw_local_anomaly_summary
            elif residual_input == "local_anomaly_profile":
                residual_source = h_raw_local_anomaly_profile
            elif residual_input == "raw_feature_topk":
                residual_source = h_raw_full
                topk = self.raw_feature_residual_topk
                if topk is None:
                    topk = self.config.classifier.residual_topk
                if topk is None:
                    topk = self.config.top_k_features if self.config.top_k_features is not None else 512
                k = min(max(1, int(topk)), residual_source.shape[-1])
                topk_idx = residual_source.abs().topk(k=k, dim=-1).indices
                residual_details["topk_indices"] = topk_idx
                residual_details["topk_values"] = residual_source.gather(1, topk_idx)
                residual_source = residual_source.gather(1, topk_idx)
            elif residual_input == "raw_feature_lowrank":
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
            residual_hidden = self.raw_feature_residual_norm(residual_source)
            if self.raw_feature_residual_lowrank_proj is not None:
                residual_hidden = self.raw_feature_residual_lowrank_proj(residual_hidden)
            if self.raw_feature_residual_activation is not None:
                residual_hidden = self.raw_feature_residual_activation(residual_hidden)
            residual = self.raw_feature_residual_proj(residual_hidden)
        residual = float(self.config.classifier.scale) * residual
        residual_max_ratio = float(getattr(self.config.classifier, "residual_max_ratio", 0.0))
        if residual_max_ratio > 0.0:
            eps = 1e-6
            core_norm = h_core.norm(dim=-1, keepdim=True).clamp_min(eps)
            residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(eps)
            max_allowed = residual_max_ratio * core_norm
            residual_ratio_scale = torch.clamp(max_allowed / residual_norm, max=1.0)
            residual = residual * residual_ratio_scale
            residual_details["residual_ratio_scale"] = residual_ratio_scale.squeeze(-1)
        residual_alignment_mode = str(getattr(self.config.classifier, "residual_alignment_mode", "off"))
        if residual_alignment_mode == "cosine":
            alignment = F.cosine_similarity(h_core, residual, dim=-1, eps=1e-6).unsqueeze(-1)
            alignment_gate = 0.5 * (alignment + 1.0)
            residual = residual * alignment_gate
            residual_details["residual_alignment_gate"] = alignment_gate.squeeze(-1)
        return h_core + residual_weight * residual, residual, residual_details

    def _extract_feature_tensors(
        self,
        x: torch.Tensor,
        file_ids: torch.Tensor | None = None,
        sampling_pass: int = 0,
        time_patch_width_override: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        x_bcl = to_bcl(x)
        sampling_mode = str(getattr(self.config, "region_sampling_mode", "global_random"))
        runtime_epoch = int(getattr(self, "runtime_epoch", 0))
        # Keep training stochastic while freezing val/test partitioning to a fixed
        # per-sample hash basis, so val_total_loss is comparable across epochs.
        base_sampling_epoch = runtime_epoch if self.training else 0
        sampling_epoch = base_sampling_epoch + max(int(sampling_pass), 0) * 9973
        sample_keys = None
        if file_ids is not None and (sampling_mode == "sample_epoch_hash" or (not self.training)):
            sample_keys = self._build_region_sampling_keys(x_bcl=x_bcl, file_ids=file_ids)
        effective_time_patch_width = (
            time_patch_width_override
            if time_patch_width_override is not None
            else self.config.time_patch_width
        )
        raw_time_patches, time_patch_starts = make_time_patches(
            x_bcl.unsqueeze(2),
            self.config.time_patch_count,
            self.config.padding_mode,
            patch_mode=self.config.time_patch_mode,
            patch_width=effective_time_patch_width,
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=int(self.config.region_sampling_seed_offset),
            return_start_indices=True,
        )
        raw_time_patches = raw_time_patches.squeeze(2)
        time_patches = self._apply_operator_bank_on_regions(raw_time_patches, self.time_operator_bank)

        raw_freq_bands, freq_band_starts = make_freq_bands(
            safe_rfft_amplitude(x_bcl).unsqueeze(2),
            self.config.freq_band_count,
            self.config.padding_mode,
            band_mode=self.config.freq_band_mode,
            band_width=self.config.freq_band_width,
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=int(self.config.region_sampling_seed_offset) + 7919,
            return_start_indices=True,
        )
        raw_freq_bands = raw_freq_bands.squeeze(2)
        freq_bands = self._apply_operator_bank_on_regions(raw_freq_bands, self.freq_operator_bank)

        time_global = self.time_operator_bank(x_bcl).unsqueeze(-2)
        freq_global = self.freq_operator_bank(safe_rfft_amplitude(x_bcl)).unsqueeze(-2)

        time_features = compute_time_indicators(time_patches, self.config.time_indicators)
        freq_features = compute_freq_indicators(freq_bands, self.config.freq_indicators)
        time_global_features = compute_time_indicators(time_global, self.config.time_indicators)
        freq_global_features = compute_freq_indicators(freq_global, self.config.freq_indicators)

        time_mask, freq_mask = self._split_feature_mask()
        time_global_mask = time_mask[:, :, :1, :]
        freq_global_mask = freq_mask[:, :, :1, :]
        time_features_masked = time_features * time_mask.to(time_features.device).unsqueeze(0)
        freq_features_masked = freq_features * freq_mask.to(freq_features.device).unsqueeze(0)
        time_global_features_masked = (
            time_global_features * time_global_mask.to(time_global_features.device).unsqueeze(0)
        )
        freq_global_features_masked = (
            freq_global_features * freq_global_mask.to(freq_global_features.device).unsqueeze(0)
        )

        h_raw_full = flatten_feature_tensors(time_features, freq_features)
        h_raw_struct_masked = flatten_feature_tensors(time_features_masked, freq_features_masked)
        h_raw_global_feature = flatten_feature_tensors(
            time_global_features_masked,
            freq_global_features_masked,
        )
        time_sem_mean = time_features_masked.mean(dim=3)
        time_sem_std = time_features_masked.std(dim=3, unbiased=False)
        time_sem_max = time_features_masked.amax(dim=3)
        time_sem_min = time_features_masked.amin(dim=3)
        time_sem_maxabs = time_features_masked.abs().amax(dim=3)
        freq_sem_mean = freq_features_masked.mean(dim=3)
        freq_sem_std = freq_features_masked.std(dim=3, unbiased=False)
        freq_sem_max = freq_features_masked.amax(dim=3)
        freq_sem_min = freq_features_masked.amin(dim=3)
        freq_sem_maxabs = freq_features_masked.abs().amax(dim=3)
        time_global_sem = time_global_features_masked.squeeze(3)
        freq_global_sem = freq_global_features_masked.squeeze(3)
        time_topk_abs = self._topk_abs_mean(
            time_features_masked,
            dim=3,
            k=self.time_summary_topk,
        )
        freq_topk_abs = self._topk_abs_mean(
            freq_features_masked,
            dim=3,
            k=self.freq_summary_topk,
        )
        time_topk_profile = self._topk_abs_values(
            time_features_masked,
            dim=3,
            k=self.time_summary_topk,
        )
        freq_topk_profile = self._topk_abs_values(
            freq_features_masked,
            dim=3,
            k=self.freq_summary_topk,
        )
        h_raw_semantic_stats = torch.cat(
            [
                time_sem_mean.flatten(1),
                time_sem_std.flatten(1),
                time_sem_max.flatten(1),
                time_sem_min.flatten(1),
                freq_sem_mean.flatten(1),
                freq_sem_std.flatten(1),
                freq_sem_max.flatten(1),
                freq_sem_min.flatten(1),
            ],
            dim=-1,
        )
        time_semantic_coord_tokens = torch.stack(
            [time_sem_mean, time_sem_std, time_sem_maxabs, time_global_sem],
            dim=-1,
        ).reshape(x_bcl.shape[0], -1, self.semantic_coord_stats_dim)
        freq_semantic_coord_tokens = torch.stack(
            [freq_sem_mean, freq_sem_std, freq_sem_maxabs, freq_global_sem],
            dim=-1,
        ).reshape(x_bcl.shape[0], -1, self.semantic_coord_stats_dim)
        h_raw_semantic_coord_tokens = torch.cat(
            [time_semantic_coord_tokens, freq_semantic_coord_tokens],
            dim=1,
        )
        h_raw_local_anomaly_summary = torch.cat(
            [
                time_sem_std.flatten(1),
                time_topk_abs.flatten(1),
                freq_sem_std.flatten(1),
                freq_topk_abs.flatten(1),
            ],
            dim=-1,
        )
        h_raw_local_anomaly_profile = torch.cat(
            [
                time_sem_std.flatten(1),
                time_topk_profile.flatten(1),
                freq_sem_std.flatten(1),
                freq_topk_profile.flatten(1),
            ],
            dim=-1,
        )

        e_t_local_flat = (
            time_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, self.config.time_patch_count, self.time_basis_dim)
        )
        e_f_local_flat = (
            freq_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, self.config.freq_band_count, self.freq_basis_dim)
        )
        e_t_global_flat = (
            time_global_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, 1, self.time_basis_dim)
        )
        e_f_global_flat = (
            freq_global_features_masked.permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(x_bcl.shape[0], self.config.in_channels, 1, self.freq_basis_dim)
        )
        return {
            "h_raw_full": h_raw_full,
            "h_raw_struct_masked": h_raw_struct_masked,
            "h_raw_global_feature": h_raw_global_feature,
            "h_raw_semantic_stats": h_raw_semantic_stats,
            "h_raw_semantic_coord_tokens": h_raw_semantic_coord_tokens,
            "h_raw_local_anomaly_summary": h_raw_local_anomaly_summary,
            "h_raw_local_anomaly_profile": h_raw_local_anomaly_profile,
            "time_features_masked": time_features_masked,
            "freq_features_masked": freq_features_masked,
            "time_global_features_masked": time_global_features_masked,
            "freq_global_features_masked": freq_global_features_masked,
            "E_t_local_flat": e_t_local_flat,
            "E_f_local_flat": e_f_local_flat,
            "E_t_global_flat": e_t_global_flat,
            "E_f_global_flat": e_f_global_flat,
            "time_patch_starts": time_patch_starts,
            "freq_band_starts": freq_band_starts,
        }

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
            "stability_consistency_penalty": zero,
            "region_ensemble_std": zero,
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

            assign_excess = torch.relu(
                (assignment_ratio - float(self.config.adaptive_proto_assignment_target)) / assign_denom
            )
            balance_excess = torch.relu(
                (balance_penalty - float(self.config.adaptive_proto_balance_target)) / balance_target
            )
            mix = float(self.config.adaptive_proto_balance_mix)
            collapse_signal = ((1.0 - mix) * assign_excess + mix * balance_excess).clamp(0.0, 1.0)
            # Negative feedback: when assignments collapse, reduce contrastive weight instead of amplifying it.
            signal = 1.0 - collapse_signal
            min_scale = float(self.config.adaptive_proto_min_scale)
            max_scale = float(self.config.adaptive_proto_max_scale)
            scale = scale * (reference.new_tensor(min_scale) + (max_scale - min_scale) * signal)

        return scale

    def _cooperative_branch_weights(
        self,
        hierarchical: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = hierarchical["time_focus"].device
        dtype = hierarchical["time_focus"].dtype
        if not self.cooperative_enabled:
            batch_size = int(hierarchical["time_focus"].shape[0])
            weights = torch.zeros((batch_size, 3), device=device, dtype=dtype)
            weights[:, 2] = 1.0
            return weights
        quality_time = (
            hierarchical["time_focus"] * hierarchical["time_agreement"]
        ).clamp_min(1e-4)
        quality_freq = (
            hierarchical["freq_focus"] * hierarchical["freq_agreement"]
        ).clamp_min(1e-4)
        quality_joint = torch.sqrt((quality_time * quality_freq).clamp_min(1e-6))
        priors = torch.tensor(
            [
                float(self.config.cooperative_time_prior),
                float(self.config.cooperative_freq_prior),
                float(self.config.cooperative_joint_prior),
            ],
            device=device,
            dtype=dtype,
        ).view(1, 3)
        quality = torch.cat([quality_time, quality_freq, quality_joint], dim=1)
        weights = quality * (priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-6))
        floor = float(self.config.cooperative_weight_floor)
        if floor > 0.0:
            uniform = torch.full_like(weights, 1.0 / float(weights.shape[1]))
            weights = (1.0 - floor) * weights + floor * uniform
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.nan_to_num(weights, nan=1.0 / float(weights.shape[1]))

    @staticmethod
    def _local_global_anomaly_weights(
        local_roles: torch.Tensor,
        global_role: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select local regions that deviate from the full-range global evidence."""
        local_norm = F.layer_norm(local_roles, (int(local_roles.shape[-1]),))
        global_norm = F.layer_norm(global_role, (int(global_role.shape[-1]),)).unsqueeze(1)
        anomaly = (local_norm - global_norm).pow(2).mean(dim=-1)
        anomaly = (anomaly - anomaly.mean(dim=1, keepdim=True)) / anomaly.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        weights = torch.softmax(anomaly, dim=1)
        return weights, anomaly

    @staticmethod
    def _combine_cross_and_anomaly_weights(
        cross_weights: torch.Tensor,
        anomaly_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Use sample-internal abnormality as the local explanation coordinate.

        Cross-consistency still builds the main local/global summary. The local
        anomaly route is different: it must point to the patch or band that
        deviates from the full-range evidence, otherwise a near-uniform cross
        distribution suppresses the very evidence this branch is meant to expose.
        """
        del cross_weights
        return anomaly_weights / anomaly_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _build_local_concept(
        g_t_local: torch.Tensor,
        g_f_local: torch.Tensor,
    ) -> torch.Tensor:
        g_t = F.layer_norm(g_t_local, (int(g_t_local.shape[-1]),))
        g_f = F.layer_norm(g_f_local, (int(g_f_local.shape[-1]),))
        return torch.cat([g_t, g_f, g_t * g_f, (g_t - g_f).abs()], dim=-1)

    @staticmethod
    def _mean_kl_to_target(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, target_probs, reduction="batchmean")

    def _run_proto_head(
        self,
        head: PrototypeHead,
        concept: torch.Tensor,
        labels: Optional[torch.Tensor],
        head_key: str,
        update_enabled: bool,
        prefix: str,
        anchor_concept: Optional[torch.Tensor] = None,
        alternative_anchor_concept: Optional[torch.Tensor] = None,
        anchor_mix_mode: str = "adaptive_margin",
        residual_logit_weight: Optional[torch.Tensor] = None,
        routing_concept: Optional[torch.Tensor] = None,
        candidate_competition: bool = False,
    ) -> Dict[str, torch.Tensor]:
        concept = torch.nan_to_num(concept)
        if anchor_concept is not None:
            anchor_concept = torch.nan_to_num(anchor_concept)
        if alternative_anchor_concept is not None:
            alternative_anchor_concept = torch.nan_to_num(alternative_anchor_concept)
        if routing_concept is not None:
            routing_concept = torch.nan_to_num(routing_concept)
        proto_out = head(
            concept,
            labels=labels,
            head_key=head_key,
            assignment_enabled=self.config.prototype_assignment_enabled,
            update_enabled=bool(update_enabled),
            anchor_h=anchor_concept,
            alternative_anchor_h=alternative_anchor_concept,
            anchor_mix_mode=anchor_mix_mode,
            residual_logit_weight=residual_logit_weight,
            routing_h=routing_concept,
            candidate_competition=bool(candidate_competition),
        )
        return {f"{prefix}_{key}": value for key, value in proto_out.items()}

    def _compute_single_head_contrastive(
        self,
        extras: Dict[str, Any],
        labels: torch.Tensor,
        prefix: str,
        epoch: Optional[int],
        total_epochs: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        reference = extras["h"]
        use_scaled_scores = bool(self.config.proto_contrastive_use_scaled_scores)
        if use_scaled_scores:
            proto_scores_key = f"{prefix}_proto_scores"
            if proto_scores_key not in extras:
                proto_scores_key = "proto_scores"
        else:
            proto_scores_key = f"{prefix}_proto_routing_scores"
            if proto_scores_key not in extras:
                proto_scores_key = (
                    "proto_routing_scores" if "proto_routing_scores" in extras else "proto_scores"
                )
        target_proto_scores_key = f"{prefix}_target_proto_scores"
        if target_proto_scores_key not in extras:
            target_proto_scores_key = "target_proto_scores"
        target_proto_scores_routed_key = f"{prefix}_target_proto_scores_routed"
        if target_proto_scores_routed_key not in extras:
            target_proto_scores_routed_key = "target_proto_scores_routed"
        target_proto_probs_key = f"{prefix}_target_proto_probs"
        if target_proto_probs_key not in extras:
            target_proto_probs_key = "target_proto_probs"
        target_class_neff_key = f"{prefix}_target_class_neff"
        if target_class_neff_key not in extras:
            target_class_neff_key = "target_class_neff"
        scheduled_proto_weight = self._scheduled_proto_weight(epoch, total_epochs)
        warmup_epochs = max(int(self.config.contrastive_warmup_epochs), 0)
        current_epoch = max(int(epoch or 0), 0)
        warmup_gate = 0.0 if current_epoch < warmup_epochs else 1.0
        proto_weight = scheduled_proto_weight * self.config.contrastive_scale
        specialization_weight = warmup_gate * float(self.config.proto_specialization_weight)
        occupancy_weight = float(self.config.prototype_occupancy_weight)

        if proto_weight <= 0.0 and specialization_weight <= 0.0 and occupancy_weight <= 0.0:
            contrastive = self._zero_contrastive_dict(reference)
            contrastive["n_eff"] = reference.new_zeros(1)
            contrastive["target_proto_probs"] = None
            contrastive["proto_positive_scores"] = reference.new_zeros(reference.shape[0])
            contrastive["proto_negative_scores"] = reference.new_zeros(reference.shape[0], 1)
            contrastive["prototype_usage"] = reference.new_zeros(
                (
                    int(extras[proto_scores_key].shape[1]),
                    int(extras[proto_scores_key].shape[2]),
                )
            )
            contrastive["adaptive_proto_scale"] = reference.new_tensor(1.0)
            contrastive["proto_lambda_schedule"] = reference.new_zeros(())
            contrastive["specialization_lambda_schedule"] = reference.new_zeros(())
            contrastive["lambda_schedule"] = reference.new_tensor(1.0)
            return contrastive

        contrastive_scores = extras[proto_scores_key]
        if use_scaled_scores:
            target_proto_scores = extras.get(target_proto_scores_key)
            target_proto_probs = extras.get(target_proto_probs_key)
        else:
            target_proto_scores = extras.get(
                target_proto_scores_routed_key,
                extras.get(target_proto_scores_key),
            )
            target_proto_probs = extras.get(target_proto_probs_key)
        target_class_neff = (
            extras.get(target_class_neff_key)
            if self.config.adaptive_contrastive_neff_enabled
            else None
        )
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
            target_class_neff=target_class_neff,
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
        adaptive_proto_scale = self._adaptive_proto_scale(contrastive)
        effective_proto_weight = reference.new_tensor(proto_weight) * adaptive_proto_scale
        effective_specialization_weight = reference.new_tensor(specialization_weight)
        total = (
            effective_proto_weight * contrastive["total"]
            + effective_specialization_weight * contrastive["specialization_penalty"]
            + occupancy_weight * contrastive["occupancy_penalty"]
        )
        contrastive["total"] = total
        contrastive["adaptive_proto_scale"] = adaptive_proto_scale
        contrastive["proto_lambda_schedule"] = effective_proto_weight
        contrastive["specialization_lambda_schedule"] = effective_specialization_weight
        contrastive["lambda_schedule"] = reference.new_tensor(1.0)
        return contrastive

    @staticmethod
    def _mean_true_vs_best_negative_margin(
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.to(scores.device).long()
        true_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)
        class_mask = F.one_hot(labels, num_classes=scores.shape[1]).to(torch.bool)
        negative_scores = scores.masked_fill(class_mask, torch.finfo(scores.dtype).min)
        best_negative = negative_scores.max(dim=1).values
        return (true_scores - best_negative).mean()

    def _attach_decision_diagnostics(
        self,
        metrics: Dict[str, torch.Tensor],
        extras: Dict[str, Any],
        labels: torch.Tensor,
    ) -> None:
        """Attach scalar decision-path diagnostics without changing the loss."""
        logits = extras.get("logits")
        if not torch.is_tensor(logits):
            return
        labels = labels.to(logits.device).long()
        final_pred = logits.argmax(dim=1)
        metrics["decision_logit_margin"] = self._mean_true_vs_best_negative_margin(
            logits,
            labels,
        )

        anchor_scores = extras.get("anchor_scores")
        if torch.is_tensor(anchor_scores):
            anchor_scores = anchor_scores.to(logits.device)
            anchor_pred = anchor_scores.argmax(dim=1)
            anchor_correct = anchor_pred.eq(labels)
            final_correct = final_pred.eq(labels)
            metrics["decision_anchor_acc"] = anchor_correct.float().mean()
            metrics["decision_anchor_margin"] = self._mean_true_vs_best_negative_margin(
                anchor_scores,
                labels,
            )
            metrics["decision_anchor_final_disagree"] = anchor_pred.ne(final_pred).float().mean()
            metrics["decision_anchor_correct_final_wrong"] = (
                anchor_correct & ~final_correct
            ).float().mean()
            metrics["decision_anchor_wrong_final_correct"] = (
                ~anchor_correct & final_correct
            ).float().mean()

        residual_scores = extras.get("pooled_residual_scores")
        if torch.is_tensor(residual_scores):
            residual_scores = residual_scores.to(logits.device)
            residual_pred = residual_scores.argmax(dim=1)
            metrics["decision_residual_acc"] = residual_pred.eq(labels).float().mean()
            metrics["decision_residual_margin"] = self._mean_true_vs_best_negative_margin(
                residual_scores,
                labels,
            )

        target_proto_probs = extras.get("target_proto_probs")
        if torch.is_tensor(target_proto_probs):
            probs = target_proto_probs.to(logits.device).clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            if probs.shape[-1] > 1:
                entropy = entropy / torch.log(
                    probs.new_tensor(float(probs.shape[-1]))
                ).clamp_min(1e-8)
            metrics["prototype_assignment_entropy"] = entropy.mean()
            metrics["prototype_assignment_top_prob"] = probs.max(dim=-1).values.mean()

        target_proto_routing_scores = extras.get("target_proto_routing_scores")
        if torch.is_tensor(target_proto_routing_scores):
            routed = target_proto_routing_scores.to(logits.device)
            if routed.shape[-1] > 1:
                top2 = routed.topk(k=2, dim=-1).values
                metrics["prototype_assignment_margin"] = (top2[:, 0] - top2[:, 1]).mean()
        for metric_name in (
            "time_focus",
            "freq_focus",
            "time_agreement",
            "freq_agreement",
            "anomaly_gate",
            "local_global_disagreement",
            "anomaly_alignment_gate",
            "h_global_deep_gate",
            "transparent_time_basis_mix",
        ):
            value = extras.get(metric_name)
            if torch.is_tensor(value):
                metrics[f"evidence_{metric_name}"] = value.to(logits.device).mean()
        anchor_path_weights = extras.get("anchor_path_weights")
        if torch.is_tensor(anchor_path_weights) and anchor_path_weights.shape[-1] >= 2:
            weights = anchor_path_weights.to(logits.device)
            metrics["decision_anchor_global_weight"] = weights[:, 0].mean()
            metrics["decision_anchor_fused_weight"] = weights[:, 1].mean()
        anchor_residual_path_weights = extras.get("anchor_residual_path_weights")
        if (
            torch.is_tensor(anchor_residual_path_weights)
            and anchor_residual_path_weights.shape[-1] >= 2
        ):
            weights = anchor_residual_path_weights.to(logits.device)
            metrics["decision_anchor_evidence_weight"] = weights[:, 0].mean()
            metrics["decision_residual_evidence_weight"] = weights[:, 1].mean()
        evidence_path_weights = extras.get("evidence_path_weights")
        if torch.is_tensor(evidence_path_weights) and evidence_path_weights.shape[-1] >= 2:
            weights = evidence_path_weights.to(logits.device)
            metrics["decision_primary_evidence_weight"] = weights[:, 0].mean()
            metrics["decision_relative_evidence_weight"] = weights[:, 1].mean()
        residual_anchor_support_gate = extras.get("residual_anchor_support_gate")
        if torch.is_tensor(residual_anchor_support_gate):
            metrics["decision_residual_anchor_support_gate"] = (
                residual_anchor_support_gate.to(logits.device).mean()
            )

    def _encode_structured(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        sampling_pass: int = 0,
        time_patch_width_override: int | None = None,
    ) -> Dict[str, Any]:
        file_ids = self._to_tensor_ids(file_ids_raw, x.device)
        feature_tensors = self._extract_feature_tensors(
            x,
            file_ids=file_ids,
            sampling_pass=sampling_pass,
            time_patch_width_override=time_patch_width_override,
        )
        h_raw_full = feature_tensors["h_raw_full"]
        h_raw_struct_masked = feature_tensors["h_raw_struct_masked"]
        h_raw_global_feature = feature_tensors["h_raw_global_feature"]
        h_raw_semantic_stats = feature_tensors["h_raw_semantic_stats"]
        h_raw_semantic_coord_tokens = feature_tensors["h_raw_semantic_coord_tokens"]
        h_raw_local_anomaly_summary = feature_tensors["h_raw_local_anomaly_summary"]
        h_raw_local_anomaly_profile = feature_tensors["h_raw_local_anomaly_profile"]
        domains = self.get_domains(file_ids)
        head_key = self.resolve_head_key(file_ids)

        e_t_local_bar, alpha_t = self.time_channel_fusion(feature_tensors["E_t_local_flat"])
        e_f_local_bar, alpha_f = self.freq_channel_fusion(feature_tensors["E_f_local_flat"])

        # Structured Evidence Dropout: during training, randomly zero out entire
        # patches (time) or bands (freq) so the model learns to diagnose from
        # partial evidence.  This improves domain generalization because the model
        # cannot over-rely on any single evidence pathway that may be unreliable
        # in an unseen domain.  Interpretation: "a robust diagnosis uses multiple
        # independent evidence pathways; the model should still work when some are
        # missing."
        if self.training and self.config.evidence_dropout_rate > 0.0:
            rate = self.config.evidence_dropout_rate
            B, P, M = e_t_local_bar.shape
            patch_keep = torch.bernoulli(
                e_t_local_bar.new_full((B, P, 1), 1.0 - rate)
            )
            e_t_local_bar = e_t_local_bar * patch_keep
            Bf, Pf, Mf = e_f_local_bar.shape
            band_keep = torch.bernoulli(
                e_f_local_bar.new_full((Bf, Pf, 1), 1.0 - rate)
            )
            e_f_local_bar = e_f_local_bar * band_keep

        # Multi-granularity: during training, randomly select ONE patch width per
        # forward pass so the model learns scale-invariant role representations.
        # During inference the primary width is used; multi-scale logit ensembling
        # is handled at the forward() level.
        if self.time_patch_widths is not None and self.training:
            all_widths = [self.config.time_patch_width] + list(self.time_patch_widths)
            pw_idx = torch.randint(0, len(all_widths), (1,)).item()
            selected_pw = all_widths[pw_idx]
            if selected_pw != self.config.time_patch_width:
                feature_tensors = self._extract_feature_tensors(
                    x,
                    file_ids=file_ids,
                    sampling_pass=sampling_pass,
                    time_patch_width_override=selected_pw,
                )
                e_t_local_bar, alpha_t = self.time_channel_fusion(feature_tensors["E_t_local_flat"])

        e_t_local_anomaly_bar = e_t_local_bar
        e_f_local_anomaly_bar = e_f_local_bar
        e_t_local_main_bar = e_t_local_bar
        e_f_local_main_bar = e_f_local_bar
        if bool(getattr(self.config, "cross_patch_norm_enabled", False)):
            scope = str(getattr(self.config, "cross_patch_norm_scope", "anomaly"))
            cp_mean_t = e_t_local_bar.mean(dim=1, keepdim=True)
            cp_std_t = e_t_local_bar.std(dim=1, keepdim=True).clamp_min(1e-6)
            e_t_local_anomaly_bar = (e_t_local_bar - cp_mean_t) / cp_std_t
            cp_mean_f = e_f_local_bar.mean(dim=1, keepdim=True)
            cp_std_f = e_f_local_bar.std(dim=1, keepdim=True).clamp_min(1e-6)
            e_f_local_anomaly_bar = (e_f_local_bar - cp_mean_f) / cp_std_f
            # Scope controls only the main local summary path.  The anomaly path
            # always uses relative evidence when enabled, so prototype
            # explanations can point to sample-internal abnormal patches/bands.
            if scope in {"time", "both"}:
                e_t_local_main_bar = e_t_local_anomaly_bar
            if scope in {"freq", "both"}:
                e_f_local_main_bar = e_f_local_anomaly_bar

        z_t_local, a_t = self.time_role_compression(e_t_local_main_bar)
        z_f_local, a_f = self.freq_role_compression(e_f_local_main_bar)
        z_t_local_anomaly, _ = self.time_role_compression(e_t_local_anomaly_bar)
        z_f_local_anomaly, _ = self.freq_role_compression(e_f_local_anomaly_bar)
        h_local_core, s, w_t_cross, w_f_cross, g_t_local_cross, g_f_local_cross = (
            self.cross_evidence_pooling(
                z_t_local,
                z_f_local,
            )
        )

        e_t_global_bar, alpha_t_global = self.time_channel_fusion(feature_tensors["E_t_global_flat"])
        e_f_global_bar, alpha_f_global = self.freq_channel_fusion(feature_tensors["E_f_global_flat"])
        z_t_global_seq, a_t_global = self.time_role_compression(e_t_global_bar)
        z_f_global_seq, a_f_global = self.freq_role_compression(e_f_global_bar)
        g_t_global_structured = z_t_global_seq.squeeze(1)
        g_f_global_structured = z_f_global_seq.squeeze(1)
        structured_global_enabled = bool(self.config.structured_global_enabled)
        g_t_global = g_t_global_structured
        g_f_global = g_f_global_structured

        h_transparent_raw = None
        h_transparent_time_raw = None
        h_transparent_freq_raw = None
        h_transparent = g_t_global.new_zeros((g_t_global.shape[0], int(self.config.concept_dim)))
        h_transparent_time = h_transparent.new_zeros(h_transparent.shape)
        h_transparent_freq = h_transparent.new_zeros(h_transparent.shape)
        h_transparent_mix = h_transparent.new_zeros(h_transparent.shape)
        transparent_mix_weights = h_transparent.new_zeros((h_transparent.shape[0], 3))
        transparent_time_role = g_t_global.new_zeros(g_t_global.shape)
        transparent_freq_role = g_f_global.new_zeros(g_f_global.shape)
        transparent_time_gate = g_t_global.new_zeros((g_t_global.shape[0], 1))
        transparent_freq_gate = g_f_global.new_zeros((g_f_global.shape[0], 1))
        transparent_time_basis_mix = g_t_global.new_zeros((g_t_global.shape[0], 1))
        if (
            self.transparent_time_backbone is not None
            and self.transparent_freq_backbone is not None
            and self.config.transparent_backbone_weight > 0.0
            and self.transparent_time_feature_norm is not None
            and self.transparent_time_feature_proj is not None
            and self.transparent_freq_feature_norm is not None
            and self.transparent_freq_feature_proj is not None
            and self.transparent_time_to_role is not None
            and self.transparent_freq_to_role is not None
        ):
            x_transparent_time = self._build_transparent_backbone_input(x, branch="time")
            x_transparent_freq = self._build_transparent_backbone_input(x, branch="freq")
            h_transparent_time_raw = self._extract_transparent_backbone_features(
                self.transparent_time_backbone,
                x_transparent_time,
            )
            h_transparent_freq_raw = self._extract_transparent_backbone_features(
                self.transparent_freq_backbone,
                x_transparent_freq,
            )
            h_transparent_raw = torch.cat([h_transparent_time_raw, h_transparent_freq_raw], dim=-1)
            h_transparent_time = self.transparent_time_feature_proj(
                self.transparent_time_feature_norm(h_transparent_time_raw)
            )
            h_transparent_freq = self.transparent_freq_feature_proj(
                self.transparent_freq_feature_norm(h_transparent_freq_raw)
            )
            transparent_time_role = self.transparent_time_to_role(h_transparent_time)
            transparent_freq_role = self.transparent_freq_to_role(h_transparent_freq)
            if (
                self.transparent_time_basis_backbone is not None
                and self.transparent_time_basis_feature_norm is not None
                and self.transparent_time_basis_feature_proj is not None
                and self.transparent_time_basis_to_role is not None
                and self.transparent_time_basis_mix_logit is not None
            ):
                h_time_basis_raw = self._extract_transparent_backbone_features(
                    self.transparent_time_basis_backbone,
                    x_transparent_time,
                )
                h_time_basis = self.transparent_time_basis_feature_proj(
                    self.transparent_time_basis_feature_norm(h_time_basis_raw)
                )
                time_basis_role = self.transparent_time_basis_to_role(h_time_basis)
                mix = torch.sigmoid(self.transparent_time_basis_mix_logit).to(
                    device=h_transparent_time.device,
                    dtype=h_transparent_time.dtype,
                )
                transparent_time_basis_mix = mix.expand(h_transparent_time.shape[0], 1)
                h_transparent_time = (
                    (1.0 - transparent_time_basis_mix) * h_transparent_time
                    + transparent_time_basis_mix * h_time_basis
                )
                transparent_time_role = (
                    (1.0 - transparent_time_basis_mix) * transparent_time_role
                    + transparent_time_basis_mix * time_basis_role
                )
            h_transparent = 0.5 * (h_transparent_time + h_transparent_freq)
            h_transparent_mix = h_transparent
            if structured_global_enabled:
                # Clean global semantic chain:
                # Use one fixed semantic coordinate for global evidence.
                # Deep-global and structured-global are fused by a static convex mix
                # to reduce seed-dependent gate oscillation.
                deep_global_weight = float(self.config.transparent_backbone_weight)
                deep_global_weight = min(max(deep_global_weight, 0.0), 1.0)
                transparent_time_blend = torch.full_like(transparent_time_gate, deep_global_weight)
                transparent_freq_blend = torch.full_like(transparent_freq_gate, deep_global_weight)
                transparent_time_gate = transparent_time_blend
                transparent_freq_gate = transparent_freq_blend
                g_t_global = F.layer_norm(
                    (1.0 - transparent_time_blend) * g_t_global_structured
                    + transparent_time_blend * transparent_time_role,
                    (int(g_t_global.shape[-1]),),
                )
                g_f_global = F.layer_norm(
                    (1.0 - transparent_freq_blend) * g_f_global_structured
                    + transparent_freq_blend * transparent_freq_role,
                    (int(g_f_global.shape[-1]),),
                )
                structured_share = torch.full_like(transparent_time_blend, 1.0 - deep_global_weight)
                transparent_mix_weights = torch.stack(
                    [
                        (0.5 * transparent_time_blend).squeeze(-1),
                        (0.5 * transparent_freq_blend).squeeze(-1),
                        structured_share.squeeze(-1),
                    ],
                    dim=1,
                )
            else:
                transparent_time_gate = torch.ones_like(transparent_time_gate)
                transparent_freq_gate = torch.ones_like(transparent_freq_gate)
                g_t_global = F.layer_norm(
                    transparent_time_role,
                    (int(g_t_global.shape[-1]),),
                )
                g_f_global = F.layer_norm(
                    transparent_freq_role,
                    (int(g_f_global.shape[-1]),),
                )
                transparent_mix_weights = torch.stack(
                    [
                        torch.full_like(transparent_time_gate.squeeze(-1), 0.5),
                        torch.full_like(transparent_freq_gate.squeeze(-1), 0.5),
                        torch.zeros_like(transparent_time_gate.squeeze(-1)),
                    ],
                    dim=1,
                )
        elif not structured_global_enabled:
            # Structured-global is disabled but deep-global branch is unavailable.
            # Fall back to structured-global to keep the main path numerically valid.
            g_t_global = g_t_global_structured
            g_f_global = g_f_global_structured
        if structured_global_enabled and h_transparent_raw is None:
            transparent_mix_weights = torch.stack(
                [
                    torch.zeros_like(transparent_time_gate.squeeze(-1)),
                    torch.zeros_like(transparent_freq_gate.squeeze(-1)),
                    torch.ones_like(transparent_time_gate.squeeze(-1)),
                ],
                dim=1,
            )

        w_t_anomaly, time_local_anomaly_logits = self._local_global_anomaly_weights(
            z_t_local_anomaly,
            g_t_global,
        )
        w_f_anomaly, freq_local_anomaly_logits = self._local_global_anomaly_weights(
            z_f_local_anomaly,
            g_f_global,
        )
        w_t_local_anomaly = self._combine_cross_and_anomaly_weights(w_t_cross, w_t_anomaly)
        w_f_local_anomaly = self._combine_cross_and_anomaly_weights(w_f_cross, w_f_anomaly)
        g_t_local_anomaly = torch.einsum("bp,bpr->br", w_t_local_anomaly, z_t_local_anomaly)
        g_f_local_anomaly = torch.einsum("bf,bfr->br", w_f_local_anomaly, z_f_local_anomaly)
        h_proto_local_raw = self._build_local_concept(g_t_local_anomaly, g_f_local_anomaly)
        w_t = w_t_cross
        w_f = w_f_cross
        g_t_local = g_t_local_cross
        g_f_local = g_f_local_cross

        h_hier_core, hierarchical = self.hierarchical_evidence_integration(
            g_t_local=g_t_local,
            g_f_local=g_f_local,
            g_t_global=g_t_global,
            g_f_global=g_f_global,
            w_t_local=w_t,
            w_f_local=w_f,
        )
        g_t = hierarchical["g_t_summary"]
        g_f = hierarchical["g_f_summary"]
        g_t = g_t + self.time_role_refiner(g_t)
        g_f = g_f + self.freq_role_refiner(g_f)
        branch_proto_mix = self._cooperative_branch_weights(hierarchical)
        g_t = torch.nan_to_num(g_t)
        g_f = torch.nan_to_num(g_f)
        # Class anchors use only the full-range deep-global evidence; stochastic local
        # regions are reserved for prototype residual/routing explanations.
        g_t_anchor = torch.nan_to_num(g_t_global + self.time_role_refiner(g_t_global))
        g_f_anchor = torch.nan_to_num(g_f_global + self.freq_role_refiner(g_f_global))
        # Fixed semantic coordinates avoid seed-sensitive learned pair projections.
        # Coordinates: time summary, frequency summary, T-F concordance, local/global gap.
        h_global_pair_raw = torch.cat([g_t, g_f], dim=-1)
        if int(self.config.concept_dim) == 4 * int(self.config.role_dim):
            h_core_base = torch.cat(
                [
                    g_t,
                    g_f,
                    F.layer_norm(g_t * g_f, (int(g_t.shape[-1]),)),
                    hierarchical["local_global_gap"],
                ],
                dim=-1,
            )
            h_anchor_base = torch.cat(
                [
                    g_t_anchor,
                    g_f_anchor,
                    F.layer_norm(g_t_anchor * g_f_anchor, (int(g_t_anchor.shape[-1]),)),
                    torch.zeros_like(g_t_anchor),
                ],
                dim=-1,
            )
        elif int(self.config.concept_dim) == 2 * int(self.config.role_dim):
            h_core_base = h_global_pair_raw
            h_anchor_base = torch.cat([g_t_anchor, g_f_anchor], dim=-1)
        else:
            raise ValueError(
                "Fixed TF concept coordinates require concept_dim to equal "
                "2 * role_dim or 4 * role_dim."
            )
        h_core_semantic = self.global_semantic_out(F.gelu(self.global_semantic_proj(h_core_base)))
        h_core_semantic = self.global_semantic_norm(h_core_base + h_core_semantic)
        h_anchor_semantic = self.global_semantic_out(F.gelu(self.global_semantic_proj(h_anchor_base)))
        h_anchor_semantic = self.global_semantic_norm(h_anchor_base + h_anchor_semantic)
        h_proto_local_semantic = self.global_semantic_out(F.gelu(self.global_semantic_proj(h_proto_local_raw)))
        h_proto_local_semantic = self.global_semantic_norm(h_proto_local_raw + h_proto_local_semantic)
        h_relative_semantic = h_core_semantic
        if bool(getattr(self.config, "cross_patch_norm_enabled", False)):
            (
                h_local_core_relative,
                _,
                w_t_relative,
                w_f_relative,
                g_t_local_relative,
                g_f_local_relative,
            ) = self.cross_evidence_pooling(
                z_t_local_anomaly,
                z_f_local_anomaly,
            )
            h_hier_relative, hierarchical_relative = self.hierarchical_evidence_integration(
                g_t_local=g_t_local_relative,
                g_f_local=g_f_local_relative,
                g_t_global=g_t_global,
                g_f_global=g_f_global,
                w_t_local=w_t_relative,
                w_f_local=w_f_relative,
            )
            g_t_relative = hierarchical_relative["g_t_summary"]
            g_f_relative = hierarchical_relative["g_f_summary"]
            g_t_relative = torch.nan_to_num(g_t_relative + self.time_role_refiner(g_t_relative))
            g_f_relative = torch.nan_to_num(g_f_relative + self.freq_role_refiner(g_f_relative))
            if int(self.config.concept_dim) == 4 * int(self.config.role_dim):
                h_relative_base = torch.cat(
                    [
                        g_t_relative,
                        g_f_relative,
                        F.layer_norm(g_t_relative * g_f_relative, (int(g_t_relative.shape[-1]),)),
                        hierarchical_relative["local_global_gap"],
                    ],
                    dim=-1,
                )
            else:
                h_relative_base = torch.cat([g_t_relative, g_f_relative], dim=-1)
            h_relative_semantic = self.global_semantic_out(
                F.gelu(self.global_semantic_proj(h_relative_base))
            )
            h_relative_semantic = self.global_semantic_norm(h_relative_base + h_relative_semantic)
        anomaly_source = h_local_core
        if self.local_anomaly_residual_mode == "off":
            anomaly_residual = torch.zeros_like(h_core_semantic)
        elif self.local_anomaly_residual_mode == "mlp":
            if self.anomaly_residual_proj is None:
                raise RuntimeError("anomaly_residual_proj is required for mlp residual mode.")
            anomaly_residual = self.anomaly_residual_proj(
                self.anomaly_residual_norm(anomaly_source)
            )
        elif self.local_anomaly_residual_mode == "direct_delta":
            if self.anomaly_residual_proj is None:
                raise RuntimeError("anomaly_residual_proj is required for direct_delta residual mode.")
            anomaly_direct = self.anomaly_residual_norm(anomaly_source)
            anomaly_delta = self.anomaly_residual_proj(anomaly_direct)
            anomaly_residual = F.layer_norm(
                anomaly_direct + anomaly_delta,
                (int(anomaly_direct.shape[-1]),),
            )
        else:
            # Direct mode keeps local abnormal evidence in the same fixed
            # T/F semantic coordinates as the global concept, avoiding a
            # seed-sensitive random projection as a second decision path.
            anomaly_residual = self.anomaly_residual_norm(anomaly_source)
        cross_focus = 0.5 * (hierarchical["time_focus"] + hierarchical["freq_focus"])
        anomaly_focus = 0.5 * (
            HierarchicalEvidenceIntegration._focus(w_t_local_anomaly)
            + HierarchicalEvidenceIntegration._focus(w_f_local_anomaly)
        )
        local_global_disagreement = hierarchical["local_global_gap"].abs().mean(dim=-1, keepdim=True)
        local_global_disagreement = local_global_disagreement / (1.0 + local_global_disagreement)
        anomaly_alignment = F.cosine_similarity(
            h_core_semantic,
            anomaly_residual,
            dim=-1,
            eps=1e-6,
        ).unsqueeze(-1)
        anomaly_alignment_gate = (0.5 * (anomaly_alignment + 1.0)).clamp(0.0, 1.0)
        anomaly_gate_mode = str(getattr(self.config, "local_anomaly_gate_mode", "focus"))
        if anomaly_gate_mode == "disagreement_aligned":
            anomaly_gate = local_global_disagreement * anomaly_alignment_gate
        elif anomaly_gate_mode == "disagreement":
            anomaly_gate = local_global_disagreement
        elif anomaly_gate_mode == "anomaly_focus":
            anomaly_gate = anomaly_focus
        else:
            anomaly_gate = cross_focus
        h_core = h_core_semantic + anomaly_gate * anomaly_residual
        h_core = F.layer_norm(h_core, (int(h_core.shape[-1]),))
        h_core = torch.nan_to_num(h_core)
        concept_source = "fixed_tf_global_local_coordinates"
        h, h_residual, residual_details = self._concept_with_residual(
            h_core,
            h_raw_full,
            h_raw_struct_masked,
            h_raw_global_feature,
            h_raw_semantic_stats,
            h_raw_semantic_coord_tokens,
            h_raw_local_anomaly_summary,
            h_raw_local_anomaly_profile,
        )
        h = torch.nan_to_num(h)
        h_global = h_core_semantic
        h_anchor_global = torch.nan_to_num(h_anchor_semantic)
        h_global_deep = h_transparent
        h_global_deep_gate = 0.5 * (transparent_time_gate + transparent_freq_gate)
        extras: Dict[str, Any] = {
            "E_t": feature_tensors["time_features_masked"],
            "E_f": feature_tensors["freq_features_masked"],
            "E_t_flat": feature_tensors["E_t_local_flat"],
            "E_f_flat": feature_tensors["E_f_local_flat"],
            "E_t_global": feature_tensors["time_global_features_masked"],
            "E_f_global": feature_tensors["freq_global_features_masked"],
            "E_t_global_flat": feature_tensors["E_t_global_flat"],
            "E_f_global_flat": feature_tensors["E_f_global_flat"],
            "alpha_t": alpha_t,
            "alpha_f": alpha_f,
            "alpha_t_global": alpha_t_global,
            "alpha_f_global": alpha_f_global,
            "E_t_bar": e_t_local_bar,
            "E_f_bar": e_f_local_bar,
            "E_t_global_bar": e_t_global_bar,
            "E_f_global_bar": e_f_global_bar,
            "E_t_global_deep_bar": e_t_global_bar,
            "E_f_global_deep_bar": e_f_global_bar,
            "E_t_global_deep_gate": transparent_time_gate,
            "E_f_global_deep_gate": transparent_freq_gate,
            "A_t": a_t,
            "A_f": a_f,
            "A_t_global": a_t_global,
            "A_f_global": a_f_global,
            "Z_t": z_t_local,
            "Z_f": z_f_local,
            "Z_t_global": z_t_global_seq,
            "Z_f_global": z_f_global_seq,
            "S": s,
            "w_t": w_t,
            "w_f": w_f,
            "w_t_cross": w_t_cross,
            "w_f_cross": w_f_cross,
            "w_t_anomaly": w_t_anomaly,
            "w_f_anomaly": w_f_anomaly,
            "w_t_local_anomaly": w_t_local_anomaly,
            "w_f_local_anomaly": w_f_local_anomaly,
            "time_local_anomaly_logits": time_local_anomaly_logits,
            "freq_local_anomaly_logits": freq_local_anomaly_logits,
            "g_t_local_cross": g_t_local_cross,
            "g_f_local_cross": g_f_local_cross,
            "g_t_local_anomaly": g_t_local_anomaly,
            "g_f_local_anomaly": g_f_local_anomaly,
            "g_t_local": g_t_local,
            "g_f_local": g_f_local,
            "g_t_global_structured": g_t_global_structured,
            "g_f_global_structured": g_f_global_structured,
            "g_t_global_deep": transparent_time_role,
            "g_f_global_deep": transparent_freq_role,
            "g_t_global_deep_gate": transparent_time_gate,
            "g_f_global_deep_gate": transparent_freq_gate,
            "g_t_global": g_t_global,
            "g_f_global": g_f_global,
            "g_t": g_t,
            "g_f": g_f,
            "time_focus": hierarchical["time_focus"],
            "freq_focus": hierarchical["freq_focus"],
            "time_agreement": hierarchical["time_agreement"],
            "freq_agreement": hierarchical["freq_agreement"],
            "time_trust_local": hierarchical["time_trust_local"],
            "freq_trust_local": hierarchical["freq_trust_local"],
            "local_global_gap": hierarchical["local_global_gap"],
            "h_local_core": h_local_core,
            "h_proto_local": h_proto_local_semantic,
            "h_proto_local_raw": h_proto_local_raw,
            "h_hier_core": h_hier_core,
            "h_core_base": h_core_base,
            "h_core_semantic": h_core_semantic,
            "h_relative": h_relative_semantic,
            "h_anchor_global": h_anchor_global,
            "h_global_pair_raw": h_global_pair_raw,
            "h_global": h_global,
            "h_global_deep": h_global_deep,
            "h_global_deep_gate": h_global_deep_gate,
            "anomaly_source": anomaly_source,
            "anomaly_residual": anomaly_residual,
            "anomaly_focus": anomaly_focus,
            "local_global_disagreement": local_global_disagreement,
            "anomaly_alignment_gate": anomaly_alignment_gate,
            "anomaly_gate": anomaly_gate,
            "prototype_evidence_weights": branch_proto_mix,
            "h_core": h_core,
            "h_residual": h_residual,
            "residual_topk_indices": residual_details["topk_indices"],
            "residual_topk_values": residual_details["topk_values"],
            "residual_semantic_coord_attention": residual_details["semantic_coord_attention"],
            "residual_ratio_scale": residual_details["residual_ratio_scale"],
            "residual_alignment_gate": residual_details["residual_alignment_gate"],
            "h_transparent_raw": h_transparent_raw,
            "h_transparent_time_raw": h_transparent_time_raw,
            "h_transparent_freq_raw": h_transparent_freq_raw,
            "h_transparent": h_transparent,
            "h_transparent_time": h_transparent_time,
            "h_transparent_freq": h_transparent_freq,
            "h_transparent_mix": h_transparent_mix,
            "transparent_time_gate": transparent_time_gate,
            "transparent_freq_gate": transparent_freq_gate,
            "transparent_mix_weights": transparent_mix_weights,
            "transparent_time_basis_mix": transparent_time_basis_mix,
            "transparent_feature_names": self.transparent_feature_names,
            "h": h,
            "c": h,
            "time_patch_starts": feature_tensors["time_patch_starts"],
            "freq_band_starts": feature_tensors["freq_band_starts"],
            "file_ids": file_ids,
            "domains": domains,
            "head_key": head_key,
            "h_raw_full": h_raw_full,
            "h_raw_struct_masked": h_raw_struct_masked,
            "h_raw": h_raw_struct_masked,
            "h_masked": h_raw_struct_masked,
            "topk_idx": None,
            "topk_mask": None,
            "variant_id": self.config.variant_id,
            "active_components": self.active_component_summary,
            "concept_source": concept_source,
        }
        return extras

    def _attach_prototype_outputs(
        self,
        encoded: Dict[str, Any],
        labels: Optional[torch.Tensor],
        update_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if update_enabled is None:
            update_enabled = bool(self.training and self.config.prototype_update_enabled)
        anchor_concept = None
        alternative_anchor_concept = None
        anchor_mix_mode = "adaptive_margin"
        prototype_anchor_input = str(getattr(self.config, "prototype_anchor_input", "global"))
        if prototype_anchor_input == "global":
            anchor_concept = encoded.get("h_anchor_global")
        elif prototype_anchor_input == "core":
            anchor_concept = encoded.get("h_core")
        elif prototype_anchor_input == "global_fused_mean":
            global_anchor = encoded.get("h_anchor_global")
            fused_anchor = encoded.get("h")
            if torch.is_tensor(global_anchor) and torch.is_tensor(fused_anchor):
                anchor_concept = F.layer_norm(
                    0.5 * (global_anchor + fused_anchor),
                    (int(global_anchor.shape[-1]),),
                )
        elif prototype_anchor_input == "dual_residual":
            anchor_concept = encoded.get("h_anchor_global")
            alternative_anchor_concept = encoded.get("h")
        elif prototype_anchor_input == "dual_residual_margin":
            anchor_concept = encoded.get("h_anchor_global")
            alternative_anchor_concept = encoded.get("h")
            anchor_mix_mode = "residual_margin"
        elif prototype_anchor_input == "dual_anchor_margin":
            anchor_concept = encoded.get("h_anchor_global")
            alternative_anchor_concept = encoded.get("h")
            anchor_mix_mode = "anchor_margin"
        elif prototype_anchor_input == "dual_mean":
            anchor_concept = encoded.get("h_anchor_global")
            alternative_anchor_concept = encoded.get("h")
            anchor_mix_mode = "mean"
        residual_logit_weight = None
        residual_logit_mode = str(getattr(self.config, "prototype_residual_logit_mode", "static"))
        if residual_logit_mode in {
            "agreement",
            "agreement_tf_balance",
            "agreement_anchor_prior",
            "agreement_anchor_uncertainty_centered",
            "agreement_local_centered",
            "agreement_candidate_centered",
            "agreement_anchor_residual_margin_mix",
            "agreement_candidate_margin_mix",
            "agreement_global_local_consensus",
            "agreement_anchor_support",
        }:
            time_agreement = encoded.get("time_agreement")
            freq_agreement = encoded.get("freq_agreement")
            if torch.is_tensor(time_agreement) and torch.is_tensor(freq_agreement):
                residual_logit_weight = torch.sqrt(
                    (time_agreement * freq_agreement).clamp_min(0.0)
                ).clamp(0.0, 1.0)
                if residual_logit_mode == "agreement_tf_balance":
                    tf_balance = (1.0 - (time_agreement - freq_agreement).abs()).clamp(0.0, 1.0)
                    residual_logit_weight = residual_logit_weight * tf_balance
                residual_logit_weight = (
                    float(getattr(self.config, "prototype_residual_logit_weight", 1.0))
                    * residual_logit_weight
                )
        candidate_competition = str(
            getattr(self.config, "prototype_residual_logit_mode", "static")
        ) in {
            "local_evidence",
            "agreement_local_centered",
            "local_competition",
            "agreement_candidate_centered",
            "agreement_candidate_margin_mix",
            "global_local_consensus",
            "agreement_global_local_consensus",
        }
        assignment_uses_local = (
            str(getattr(self.config, "prototype_assignment_input", "concept"))
            == "local_anomaly"
        )
        routing_concept = (
            encoded.get("h_proto_local")
            if (candidate_competition or assignment_uses_local)
            else None
        )
        prototype_concept_input = str(getattr(self.config, "prototype_concept_input", "h"))
        if prototype_concept_input == "local_anomaly":
            prototype_concept = encoded.get("h_proto_local", encoded["h"])
        elif prototype_concept_input == "relative":
            prototype_concept = encoded.get("h_relative", encoded["h"])
        elif prototype_concept_input == "core":
            prototype_concept = encoded.get("h_core", encoded["h"])
        else:
            prototype_concept = encoded["h"]
        joint_proto_out = self._run_proto_head(
            self.joint_prototype_head,
            prototype_concept,
            labels=labels,
            head_key=encoded["head_key"],
            update_enabled=bool(update_enabled),
            prefix="joint",
            anchor_concept=anchor_concept,
            alternative_anchor_concept=alternative_anchor_concept,
            anchor_mix_mode=anchor_mix_mode,
            residual_logit_weight=residual_logit_weight,
            routing_concept=routing_concept,
            candidate_competition=candidate_competition,
        )
        relative_proto_out: Optional[Dict[str, torch.Tensor]] = None
        evidence_path_weights = None
        fused_joint_logits = joint_proto_out["joint_logits"]
        if prototype_concept_input in {"dual_relative", "dual_relative_learned"}:
            relative_concept = encoded.get("h_relative")
            primary_concept = encoded.get("h")
            if torch.is_tensor(relative_concept) and torch.is_tensor(primary_concept):
                relative_proto_out = self._run_proto_head(
                    self.joint_prototype_head,
                    relative_concept,
                    labels=labels,
                    head_key=encoded["head_key"],
                    update_enabled=False,
                    prefix="relative",
                    anchor_concept=anchor_concept,
                    alternative_anchor_concept=None,
                    anchor_mix_mode=anchor_mix_mode,
                    residual_logit_weight=residual_logit_weight,
                    routing_concept=None,
                    candidate_competition=False,
                )
                primary_logits = joint_proto_out["joint_logits"]
                relative_logits = relative_proto_out["relative_logits"]
                if primary_logits.shape[1] <= 1:
                    evidence_path_weights = primary_logits.new_zeros((primary_logits.shape[0], 2))
                    evidence_path_weights[:, 0] = 1.0
                elif (
                    prototype_concept_input == "dual_relative_learned"
                    and self.evidence_path_gate_weight is not None
                ):
                    primary_top2 = primary_logits.topk(k=2, dim=1).values
                    relative_top2 = relative_logits.topk(k=2, dim=1).values
                    primary_margin = primary_top2[:, 0] - primary_top2[:, 1]
                    relative_margin = relative_top2[:, 0] - relative_top2[:, 1]
                    time_agreement = encoded.get("time_agreement")
                    freq_agreement = encoded.get("freq_agreement")
                    local_global_gap = encoded.get("local_global_disagreement")
                    if not torch.is_tensor(time_agreement):
                        time_agreement = primary_margin.new_zeros(primary_margin.shape[0], 1)
                    if not torch.is_tensor(freq_agreement):
                        freq_agreement = primary_margin.new_zeros(primary_margin.shape[0], 1)
                    if not torch.is_tensor(local_global_gap):
                        local_global_gap = primary_margin.new_zeros(primary_margin.shape[0], 1)
                    tf_agreement = torch.sqrt(
                        (
                            time_agreement.to(primary_logits.device, primary_logits.dtype)
                            * freq_agreement.to(primary_logits.device, primary_logits.dtype)
                        ).clamp_min(0.0)
                    )
                    gate_features = torch.cat(
                        [
                            primary_margin.unsqueeze(1),
                            relative_margin.unsqueeze(1),
                            tf_agreement.view(-1, 1),
                            local_global_gap.to(primary_logits.device, primary_logits.dtype).view(-1, 1),
                        ],
                        dim=1,
                    )
                    gate_features = F.layer_norm(gate_features, (int(gate_features.shape[-1]),))
                    gate_logits = gate_features @ self.evidence_path_gate_weight.to(
                        device=gate_features.device,
                        dtype=gate_features.dtype,
                    )
                    evidence_path_weights = torch.softmax(gate_logits, dim=1)
                else:
                    primary_margin = primary_logits.topk(k=2, dim=1).values
                    primary_margin = primary_margin[:, 0] - primary_margin[:, 1]
                    relative_margin = relative_logits.topk(k=2, dim=1).values
                    relative_margin = relative_margin[:, 0] - relative_margin[:, 1]
                    evidence_path_weights = torch.softmax(
                        torch.stack([primary_margin, relative_margin], dim=1),
                        dim=1,
                    )
                fused_joint_logits = (
                    evidence_path_weights[:, 0:1] * primary_logits
                    + evidence_path_weights[:, 1:2] * relative_logits
                )
        if not self.cooperative_enabled:
            logits = torch.nan_to_num(fused_joint_logits)
            batch_size = int(logits.shape[0])
            prototype_mix = logits.new_zeros((batch_size, 3))
            prototype_mix[:, 2] = 1.0
            extras = dict(encoded)
            extras.update(
                {
                    "logits": logits,
                    "fused_probs": torch.softmax(logits, dim=-1),
                    "joint_logits": joint_proto_out["joint_logits"],
                    "joint_logits_calibrated": fused_joint_logits,
                    "relative_logits": (
                        relative_proto_out["relative_logits"]
                        if relative_proto_out is not None
                        else None
                    ),
                    "evidence_path_weights": evidence_path_weights,
                    "joint_anchor_scores": joint_proto_out["joint_anchor_scores"],
                    "joint_anchor_scores_raw": joint_proto_out["joint_anchor_scores_raw"],
                    "joint_class_anchor": joint_proto_out["joint_class_anchor"],
                    "prototype_mix_weights": prototype_mix,
                    "prototype_confidence_weights": prototype_mix,
                    "prototype_agreement_weights": prototype_mix,
                    "prototype_reliability_weights": prototype_mix,
                    "prototype_head_logit_scales": torch.ones_like(prototype_mix),
                    "prototype_head_confidence": prototype_mix,
                    "prototype_head_agreement": prototype_mix,
                    "joint_proto_scores": joint_proto_out["joint_proto_scores"],
                    "joint_proto_scores_logits": joint_proto_out["joint_proto_scores_logits"],
                    "joint_proto_scores_raw": joint_proto_out["joint_proto_scores_raw"],
                    "joint_proto_residual_scores_raw": joint_proto_out["joint_proto_residual_scores_raw"],
                    "joint_pooled_residual_scores": joint_proto_out["joint_pooled_residual_scores"],
                    "joint_anchor_path_weights": joint_proto_out["joint_anchor_path_weights"],
                    "joint_anchor_residual_path_weights": joint_proto_out[
                        "joint_anchor_residual_path_weights"
                    ],
                    "joint_residual_anchor_support_gate": joint_proto_out[
                        "joint_residual_anchor_support_gate"
                    ],
                    "joint_proto_routing_scores": joint_proto_out["joint_proto_routing_scores"],
                    "joint_local_proto_routing_scores": joint_proto_out[
                        "joint_local_proto_routing_scores"
                    ],
                    "joint_local_proto_pool_weights": joint_proto_out[
                        "joint_local_proto_pool_weights"
                    ],
                    "joint_prototype_candidate_mask": joint_proto_out[
                        "joint_prototype_candidate_mask"
                    ],
                    "joint_target_class_ids": joint_proto_out["joint_target_class_ids"],
                    "joint_class_neff": joint_proto_out["joint_class_neff"],
                    "joint_target_class_neff": joint_proto_out["joint_target_class_neff"],
                    "joint_class_temperatures": joint_proto_out["joint_class_temperatures"],
                    "joint_target_class_temperatures": joint_proto_out["joint_target_class_temperatures"],
                    "joint_class_effective_k_bias": joint_proto_out["joint_class_effective_k_bias"],
                    "joint_class_effective_k": joint_proto_out["joint_class_effective_k"],
                    "joint_target_class_effective_k": joint_proto_out["joint_target_class_effective_k"],
                    "joint_target_proto_scores": joint_proto_out["joint_target_proto_scores"],
                    "joint_target_proto_scores_raw": joint_proto_out["joint_target_proto_scores_raw"],
                    "joint_target_proto_routing_scores": joint_proto_out["joint_target_proto_routing_scores"],
                    "joint_target_proto_scores_routed": joint_proto_out["joint_target_proto_scores_routed"],
                    "joint_target_proto_probs": joint_proto_out["joint_target_proto_probs"],
                    "joint_prototype_assignments": joint_proto_out["joint_prototype_assignments"],
                    "joint_prototype_assignment_weights": joint_proto_out["joint_prototype_assignment_weights"],
                    "joint_prototype_pos_scores": joint_proto_out["joint_prototype_pos_scores"],
                    "proto_scores": joint_proto_out["joint_proto_scores"],
                    "proto_scores_logits": joint_proto_out["joint_proto_scores_logits"],
                    "proto_scores_raw": joint_proto_out["joint_proto_scores_raw"],
                    "anchor_scores": joint_proto_out["joint_anchor_scores"],
                    "anchor_scores_raw": joint_proto_out["joint_anchor_scores_raw"],
                    "class_anchor": joint_proto_out["joint_class_anchor"],
                    "proto_residual_scores_raw": joint_proto_out["joint_proto_residual_scores_raw"],
                    "pooled_residual_scores": joint_proto_out["joint_pooled_residual_scores"],
                    "anchor_path_weights": joint_proto_out["joint_anchor_path_weights"],
                    "anchor_residual_path_weights": joint_proto_out[
                        "joint_anchor_residual_path_weights"
                    ],
                    "residual_anchor_support_gate": joint_proto_out[
                        "joint_residual_anchor_support_gate"
                    ],
                    "proto_routing_scores": joint_proto_out["joint_proto_routing_scores"],
                    "local_proto_routing_scores": joint_proto_out[
                        "joint_local_proto_routing_scores"
                    ],
                    "local_proto_pool_weights": joint_proto_out["joint_local_proto_pool_weights"],
                    "prototype_candidate_mask": joint_proto_out["joint_prototype_candidate_mask"],
                    "target_class_ids": joint_proto_out["joint_target_class_ids"],
                    "class_neff": joint_proto_out["joint_class_neff"],
                    "target_class_neff": joint_proto_out["joint_target_class_neff"],
                    "class_temperatures": joint_proto_out["joint_class_temperatures"],
                    "target_class_temperatures": joint_proto_out["joint_target_class_temperatures"],
                    "class_effective_k_bias": joint_proto_out["joint_class_effective_k_bias"],
                    "class_effective_k": joint_proto_out["joint_class_effective_k"],
                    "target_class_effective_k": joint_proto_out["joint_target_class_effective_k"],
                    "target_proto_scores": joint_proto_out["joint_target_proto_scores"],
                    "target_proto_scores_raw": joint_proto_out["joint_target_proto_scores_raw"],
                    "target_proto_routing_scores": joint_proto_out["joint_target_proto_routing_scores"],
                    "target_proto_scores_routed": joint_proto_out["joint_target_proto_scores_routed"],
                    "target_proto_probs": joint_proto_out["joint_target_proto_probs"],
                    "prototype_assignments": joint_proto_out["joint_prototype_assignments"],
                    "prototype_assignment_weights": joint_proto_out["joint_prototype_assignment_weights"],
                    "prototype_pos_scores": joint_proto_out["joint_prototype_pos_scores"],
                    "labels": labels,
                }
            )
            return extras
        if self.time_prototype_head is None or self.freq_prototype_head is None:
            raise RuntimeError(
                "Cooperative prototype mode is enabled but time/freq prototype heads are not initialized."
            )
        time_proto_out = self._run_proto_head(
            self.time_prototype_head,
            encoded["g_t"],
            labels=labels,
            head_key=encoded["head_key"],
            update_enabled=bool(update_enabled),
            prefix="time",
        )
        freq_proto_out = self._run_proto_head(
            self.freq_prototype_head,
            encoded["g_f"],
            labels=labels,
            head_key=encoded["head_key"],
            update_enabled=bool(update_enabled),
            prefix="freq",
        )
        head_logits = torch.stack(
            [
                time_proto_out["time_logits"],
                freq_proto_out["freq_logits"],
                joint_proto_out["joint_logits"],
            ],
            dim=1,
        )
        if self.cooperative_prototype_fusion is None:
            raise RuntimeError("cooperative_prototype_fusion is not initialized while cooperative mode is enabled.")
        fusion = self.cooperative_prototype_fusion(
            encoded["prototype_evidence_weights"],
            head_logits,
        )
        logits = torch.nan_to_num(fusion["fused_logits"])
        extras = dict(encoded)
        extras.update(
            {
                "logits": logits,
                "fused_probs": fusion["fused_probs"],
                "time_logits": time_proto_out["time_logits"],
                "freq_logits": freq_proto_out["freq_logits"],
                "joint_logits": joint_proto_out["joint_logits"],
                "time_anchor_scores": time_proto_out["time_anchor_scores"],
                "freq_anchor_scores": freq_proto_out["freq_anchor_scores"],
                "joint_anchor_scores": joint_proto_out["joint_anchor_scores"],
                "time_anchor_scores_raw": time_proto_out["time_anchor_scores_raw"],
                "freq_anchor_scores_raw": freq_proto_out["freq_anchor_scores_raw"],
                "joint_anchor_scores_raw": joint_proto_out["joint_anchor_scores_raw"],
                "time_class_anchor": time_proto_out["time_class_anchor"],
                "freq_class_anchor": freq_proto_out["freq_class_anchor"],
                "joint_class_anchor": joint_proto_out["joint_class_anchor"],
                "time_logits_calibrated": fusion["calibrated_head_logits"][:, 0],
                "freq_logits_calibrated": fusion["calibrated_head_logits"][:, 1],
                "joint_logits_calibrated": fusion["calibrated_head_logits"][:, 2],
                "prototype_mix_weights": fusion["weights"],
                "prototype_confidence_weights": fusion["confidence_weights"],
                "prototype_agreement_weights": fusion["agreement_weights"],
                "prototype_reliability_weights": fusion["reliability_weights"],
                "prototype_head_logit_scales": fusion["head_logit_scales"],
                "prototype_head_confidence": fusion["head_confidence"],
                "prototype_head_agreement": fusion["head_agreement"],
                "time_proto_scores": time_proto_out["time_proto_scores"],
                "freq_proto_scores": freq_proto_out["freq_proto_scores"],
                "joint_proto_scores": joint_proto_out["joint_proto_scores"],
                "time_proto_scores_logits": time_proto_out["time_proto_scores_logits"],
                "freq_proto_scores_logits": freq_proto_out["freq_proto_scores_logits"],
                "joint_proto_scores_logits": joint_proto_out["joint_proto_scores_logits"],
                "time_proto_scores_raw": time_proto_out["time_proto_scores_raw"],
                "freq_proto_scores_raw": freq_proto_out["freq_proto_scores_raw"],
                "joint_proto_scores_raw": joint_proto_out["joint_proto_scores_raw"],
                "time_proto_residual_scores_raw": time_proto_out["time_proto_residual_scores_raw"],
                "freq_proto_residual_scores_raw": freq_proto_out["freq_proto_residual_scores_raw"],
                "joint_proto_residual_scores_raw": joint_proto_out["joint_proto_residual_scores_raw"],
                "time_proto_routing_scores": time_proto_out["time_proto_routing_scores"],
                "freq_proto_routing_scores": freq_proto_out["freq_proto_routing_scores"],
                "joint_proto_routing_scores": joint_proto_out["joint_proto_routing_scores"],
                "time_target_class_ids": time_proto_out["time_target_class_ids"],
                "freq_target_class_ids": freq_proto_out["freq_target_class_ids"],
                "joint_target_class_ids": joint_proto_out["joint_target_class_ids"],
                "time_class_neff": time_proto_out["time_class_neff"],
                "freq_class_neff": freq_proto_out["freq_class_neff"],
                "joint_class_neff": joint_proto_out["joint_class_neff"],
                "time_target_class_neff": time_proto_out["time_target_class_neff"],
                "freq_target_class_neff": freq_proto_out["freq_target_class_neff"],
                "joint_target_class_neff": joint_proto_out["joint_target_class_neff"],
                "time_class_temperatures": time_proto_out["time_class_temperatures"],
                "freq_class_temperatures": freq_proto_out["freq_class_temperatures"],
                "joint_class_temperatures": joint_proto_out["joint_class_temperatures"],
                "time_target_class_temperatures": time_proto_out["time_target_class_temperatures"],
                "freq_target_class_temperatures": freq_proto_out["freq_target_class_temperatures"],
                "joint_target_class_temperatures": joint_proto_out["joint_target_class_temperatures"],
                "time_class_effective_k_bias": time_proto_out["time_class_effective_k_bias"],
                "freq_class_effective_k_bias": freq_proto_out["freq_class_effective_k_bias"],
                "joint_class_effective_k_bias": joint_proto_out["joint_class_effective_k_bias"],
                "time_class_effective_k": time_proto_out["time_class_effective_k"],
                "freq_class_effective_k": freq_proto_out["freq_class_effective_k"],
                "joint_class_effective_k": joint_proto_out["joint_class_effective_k"],
                "time_target_class_effective_k": time_proto_out["time_target_class_effective_k"],
                "freq_target_class_effective_k": freq_proto_out["freq_target_class_effective_k"],
                "joint_target_class_effective_k": joint_proto_out["joint_target_class_effective_k"],
                "time_target_proto_scores": time_proto_out["time_target_proto_scores"],
                "freq_target_proto_scores": freq_proto_out["freq_target_proto_scores"],
                "joint_target_proto_scores": joint_proto_out["joint_target_proto_scores"],
                "time_target_proto_scores_raw": time_proto_out["time_target_proto_scores_raw"],
                "freq_target_proto_scores_raw": freq_proto_out["freq_target_proto_scores_raw"],
                "joint_target_proto_scores_raw": joint_proto_out["joint_target_proto_scores_raw"],
                "time_target_proto_routing_scores": time_proto_out["time_target_proto_routing_scores"],
                "freq_target_proto_routing_scores": freq_proto_out["freq_target_proto_routing_scores"],
                "joint_target_proto_routing_scores": joint_proto_out["joint_target_proto_routing_scores"],
                "time_target_proto_scores_routed": time_proto_out["time_target_proto_scores_routed"],
                "freq_target_proto_scores_routed": freq_proto_out["freq_target_proto_scores_routed"],
                "joint_target_proto_scores_routed": joint_proto_out["joint_target_proto_scores_routed"],
                "time_target_proto_probs": time_proto_out["time_target_proto_probs"],
                "freq_target_proto_probs": freq_proto_out["freq_target_proto_probs"],
                "joint_target_proto_probs": joint_proto_out["joint_target_proto_probs"],
                "time_prototype_assignments": time_proto_out["time_prototype_assignments"],
                "freq_prototype_assignments": freq_proto_out["freq_prototype_assignments"],
                "joint_prototype_assignments": joint_proto_out["joint_prototype_assignments"],
                "time_prototype_assignment_weights": time_proto_out["time_prototype_assignment_weights"],
                "freq_prototype_assignment_weights": freq_proto_out["freq_prototype_assignment_weights"],
                "joint_prototype_assignment_weights": joint_proto_out["joint_prototype_assignment_weights"],
                "time_prototype_pos_scores": time_proto_out["time_prototype_pos_scores"],
                "freq_prototype_pos_scores": freq_proto_out["freq_prototype_pos_scores"],
                "joint_prototype_pos_scores": joint_proto_out["joint_prototype_pos_scores"],
                "proto_scores": joint_proto_out["joint_proto_scores"],
                "proto_scores_logits": joint_proto_out["joint_proto_scores_logits"],
                "proto_scores_raw": joint_proto_out["joint_proto_scores_raw"],
                "anchor_scores": joint_proto_out["joint_anchor_scores"],
                "anchor_scores_raw": joint_proto_out["joint_anchor_scores_raw"],
                "class_anchor": joint_proto_out["joint_class_anchor"],
                "proto_residual_scores_raw": joint_proto_out["joint_proto_residual_scores_raw"],
                "proto_routing_scores": joint_proto_out["joint_proto_routing_scores"],
                "target_class_ids": joint_proto_out["joint_target_class_ids"],
                "class_neff": joint_proto_out["joint_class_neff"],
                "target_class_neff": joint_proto_out["joint_target_class_neff"],
                "class_temperatures": joint_proto_out["joint_class_temperatures"],
                "target_class_temperatures": joint_proto_out["joint_target_class_temperatures"],
                "class_effective_k_bias": joint_proto_out["joint_class_effective_k_bias"],
                "class_effective_k": joint_proto_out["joint_class_effective_k"],
                "target_class_effective_k": joint_proto_out["joint_target_class_effective_k"],
                "target_proto_scores": joint_proto_out["joint_target_proto_scores"],
                "target_proto_scores_raw": joint_proto_out["joint_target_proto_scores_raw"],
                "target_proto_routing_scores": joint_proto_out["joint_target_proto_routing_scores"],
                "target_proto_scores_routed": joint_proto_out["joint_target_proto_scores_routed"],
                "target_proto_probs": joint_proto_out["joint_target_proto_probs"],
                "prototype_assignments": joint_proto_out["joint_prototype_assignments"],
                "prototype_assignment_weights": joint_proto_out["joint_prototype_assignment_weights"],
                "prototype_pos_scores": joint_proto_out["joint_prototype_pos_scores"],
                "labels": labels,
            }
        )
        return extras

    def _merge_encoded_structured(self, encoded_outputs: list[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(encoded_outputs[0])
        for key, value in list(merged.items()):
            if not torch.is_tensor(value) or not value.dtype.is_floating_point:
                continue
            stacked = [output.get(key) for output in encoded_outputs]
            if any(item is None for item in stacked):
                continue
            merged[key] = torch.stack(stacked, dim=0).mean(dim=0)
        h_stack = torch.stack([output["h"] for output in encoded_outputs], dim=0)
        merged["region_ensemble_h_samples"] = h_stack
        merged["stability_consistency_penalty"] = (h_stack - h_stack.mean(dim=0, keepdim=True)).pow(2).mean()
        merged["region_ensemble_std"] = h_stack.std(dim=0, unbiased=False).mean()
        return merged

    def _forward_structured(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
        sampling_pass: int = 0,
        update_enabled: Optional[bool] = None,
        time_patch_width_override: int | None = None,
    ) -> Dict[str, Any]:
        encoded = self._encode_structured(
            x=x,
            file_ids_raw=file_ids_raw,
            sampling_pass=sampling_pass,
            time_patch_width_override=time_patch_width_override,
        )
        return self._attach_prototype_outputs(encoded, labels=labels, update_enabled=update_enabled)

    def _forward_structured_ensemble(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
        sampling_pass: int = 0,
    ) -> Dict[str, Any]:
        sample_count = max(int(self.config.region_ensemble_samples), 1)
        encoded_outputs = [
            self._encode_structured(
                x=x,
                file_ids_raw=file_ids_raw,
                sampling_pass=sampling_pass + sample_index,
            )
            for sample_index in range(sample_count)
        ]
        merged = self._merge_encoded_structured(encoded_outputs)
        return self._attach_prototype_outputs(
            merged,
            labels=labels,
            update_enabled=bool(self.training and self.config.prototype_update_enabled),
        )

    def _forward_once(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
        sampling_pass: int = 0,
        time_patch_width_override: int | None = None,
    ) -> Dict[str, Any]:
        if self._should_use_region_ensemble():
            return self._forward_structured_ensemble(
                x=x,
                file_ids_raw=file_ids_raw,
                labels=labels,
                sampling_pass=sampling_pass,
            )
        return self._forward_structured(
            x=x,
            file_ids_raw=file_ids_raw,
            labels=labels,
            sampling_pass=sampling_pass,
            time_patch_width_override=time_patch_width_override,
        )

    def _forward_eval_mc(
        self,
        x: torch.Tensor,
        file_ids_raw: Any,
        labels: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        sample_count = max(int(self.config.eval_mc_samples), 1)
        outputs = [
            self._forward_once(
                x=x,
                file_ids_raw=file_ids_raw,
                labels=labels,
                sampling_pass=sample_index,
            )
            for sample_index in range(sample_count)
        ]
        if sample_count == 1:
            return outputs[0]

        merged = dict(outputs[0])
        for key, value in list(merged.items()):
            if not torch.is_tensor(value) or not value.dtype.is_floating_point:
                continue
            stacked_values = [output.get(key) for output in outputs]
            if any(item is None or not torch.is_tensor(item) for item in stacked_values):
                continue
            if any(item.shape != value.shape for item in stacked_values):
                continue
            merged[key] = torch.stack(stacked_values, dim=0).mean(dim=0)

        logits_stack = torch.stack([output["logits"] for output in outputs], dim=0)
        merged["logits_mc_samples"] = logits_stack
        merged["logits"] = logits_stack.mean(dim=0)
        if "h" in merged and torch.is_tensor(merged["h"]):
            h_stack = torch.stack([output["h"] for output in outputs], dim=0)
            merged["region_ensemble_h_samples"] = h_stack
            merged["region_ensemble_std"] = h_stack.std(dim=0, unbiased=False).mean()
            merged["stability_consistency_penalty"] = (
                h_stack - h_stack.mean(dim=0, keepdim=True)
            ).pow(2).mean()
        return merged

    def forward_with_batch(self, batch: Dict[str, Any], epoch: int = 0) -> Dict[str, Any]:
        labels = batch.get("y")
        labels_tensor = labels.to(self.feature_mask.device).long() if torch.is_tensor(labels) else None
        x = batch["x"]
        file_ids_raw = batch.get("_file_ids_raw", batch.get("file_id"))
        if self._should_use_eval_mc():
            return self._forward_eval_mc(
                x=x,
                file_ids_raw=file_ids_raw,
                labels=labels_tensor,
            )
        return self._forward_once(
            x=x,
            file_ids_raw=file_ids_raw,
            labels=labels_tensor,
        )

    def forward(self, x: torch.Tensor, file_id=None, task_id=None) -> torch.Tensor:
        extras = self._forward_once(x=x, file_ids_raw=file_id, labels=None)
        # Multi-scale logit ensemble at inference: average predictions from
        # all configured patch widths for more robust cross-domain decisions.
        if self.time_patch_widths is not None and not self.training:
            all_logits = [extras["logits"]]
            for pw in self.time_patch_widths:
                ex_pw = self._forward_once(
                    x=x,
                    file_ids_raw=file_id,
                    labels=None,
                    time_patch_width_override=pw,
                )
                all_logits.append(ex_pw["logits"])
            return torch.stack(all_logits, dim=0).mean(dim=0)
        return extras["logits"]

    def compute_contrastive_loss(
        self,
        extras: Dict[str, Any],
        labels: torch.Tensor,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        consistency_weight = float(self.config.stability_consistency_weight)
        consistency_penalty = extras.get("stability_consistency_penalty", extras["h"].new_zeros(()))
        region_ensemble_std = extras.get("region_ensemble_std", extras["h"].new_zeros(()))
        consensus_weight = float(self.config.cooperative_consensus_weight)
        if not self.config.use_contrastive_head and consistency_weight <= 0.0 and consensus_weight <= 0.0:
            return self._zero_contrastive_dict(extras["h"])
        if not self.cooperative_enabled:
            joint_contrastive = self._compute_single_head_contrastive(
                extras, labels, prefix="joint", epoch=epoch, total_epochs=total_epochs
            )
            diversity_weight = float(self.config.prototype_diversity_weight)
            if diversity_weight > 0.0:
                diversity_penalty = self.joint_prototype_head.diversity_penalty(extras["head_key"])
                diversity_scale = extras["h"].new_tensor(1.0)
            else:
                diversity_penalty = extras["h"].new_zeros(())
                diversity_scale = extras["h"].new_tensor(1.0)
            total = joint_contrastive["total"] + diversity_weight * diversity_scale * diversity_penalty
            if consistency_weight > 0.0:
                total = total + consistency_weight * consistency_penalty
            joint_contrastive["total"] = total
            joint_contrastive["prototype_diversity_penalty"] = diversity_penalty
            joint_contrastive["prototype_diversity_scale"] = diversity_scale
            joint_contrastive["prototype_mix_time"] = extras["h"].new_zeros(())
            joint_contrastive["prototype_mix_freq"] = extras["h"].new_zeros(())
            joint_contrastive["prototype_mix_joint"] = extras["h"].new_tensor(1.0)
            joint_contrastive["prototype_consensus_penalty"] = extras["h"].new_zeros(())
            joint_contrastive["stability_consistency_penalty"] = consistency_penalty
            joint_contrastive["region_ensemble_std"] = region_ensemble_std
            extras["target_proto_probs"] = extras.get("joint_target_proto_probs", extras.get("target_proto_probs"))
            extras["proto_positive_scores"] = joint_contrastive["proto_positive_scores"]
            extras["proto_negative_scores"] = joint_contrastive["proto_negative_scores"]
            extras["prototype_usage"] = joint_contrastive["prototype_usage"]
            self._attach_decision_diagnostics(joint_contrastive, extras, labels)
            return joint_contrastive
        if self.time_prototype_head is None or self.freq_prototype_head is None:
            raise RuntimeError(
                "Cooperative contrastive path is enabled but time/freq prototype heads are not initialized."
            )

        time_contrastive = self._compute_single_head_contrastive(
            extras, labels, prefix="time", epoch=epoch, total_epochs=total_epochs
        )
        freq_contrastive = self._compute_single_head_contrastive(
            extras, labels, prefix="freq", epoch=epoch, total_epochs=total_epochs
        )
        joint_contrastive = self._compute_single_head_contrastive(
            extras, labels, prefix="joint", epoch=epoch, total_epochs=total_epochs
        )
        mix_weights = extras.get("prototype_mix_weights")
        if mix_weights is None:
            mix_weights = extras["h"].new_full((extras["h"].shape[0], 3), 1.0 / 3.0)
        mix_weights = mix_weights.detach()
        mean_mix = mix_weights.mean(dim=0)
        total = (
            mean_mix[0] * time_contrastive["total"]
            + mean_mix[1] * freq_contrastive["total"]
            + mean_mix[2] * joint_contrastive["total"]
        )
        diversity_weight = float(self.config.prototype_diversity_weight)
        if diversity_weight > 0.0:
            time_div = self.time_prototype_head.diversity_penalty(extras["head_key"])
            freq_div = self.freq_prototype_head.diversity_penalty(extras["head_key"])
            joint_div = self.joint_prototype_head.diversity_penalty(extras["head_key"])
            diversity_penalty = mean_mix[0] * time_div + mean_mix[1] * freq_div + mean_mix[2] * joint_div
            diversity_scale = extras["h"].new_tensor(1.0)
        else:
            diversity_penalty = extras["h"].new_zeros(())
            diversity_scale = extras["h"].new_tensor(1.0)
        total = total + diversity_weight * diversity_scale * diversity_penalty
        if consensus_weight > 0.0:
            target_probs = extras.get("fused_probs")
            if target_probs is None:
                target_probs = torch.softmax(extras["logits"].detach(), dim=-1)
            else:
                target_probs = target_probs.detach()
            time_logits = extras.get("time_logits_calibrated", extras.get("time_logits", extras["logits"]))
            freq_logits = extras.get("freq_logits_calibrated", extras.get("freq_logits", extras["logits"]))
            mix_weights = extras.get("prototype_mix_weights")
            if mix_weights is None:
                aux_mix = extras["h"].new_full((2,), 0.5)
            else:
                aux_mix = mix_weights[:, :2].mean(dim=0).detach()
                aux_mix = aux_mix / aux_mix.sum().clamp_min(1e-6)
            consensus_penalty = (
                aux_mix[0] * self._mean_kl_to_target(time_logits, target_probs)
                + aux_mix[1] * self._mean_kl_to_target(freq_logits, target_probs)
            )
            total = total + consensus_weight * consensus_penalty
        else:
            consensus_penalty = extras["h"].new_zeros(())
        if consistency_weight > 0.0:
            total = total + consistency_weight * consistency_penalty
        contrastive = self._zero_contrastive_dict(extras["h"])
        contrastive["total"] = total
        contrastive["lambda_schedule"] = extras["h"].new_tensor(1.0)
        contrastive["adaptive_proto_scale"] = (
            mean_mix[0] * time_contrastive["adaptive_proto_scale"]
            + mean_mix[1] * freq_contrastive["adaptive_proto_scale"]
            + mean_mix[2] * joint_contrastive["adaptive_proto_scale"]
        )
        contrastive["proto_nce"] = (
            mean_mix[0] * time_contrastive["proto_nce"]
            + mean_mix[1] * freq_contrastive["proto_nce"]
            + mean_mix[2] * joint_contrastive["proto_nce"]
        )
        contrastive["prototype_contrastive"] = contrastive["proto_nce"]
        contrastive["balance_penalty"] = (
            mean_mix[0] * time_contrastive["balance_penalty"]
            + mean_mix[1] * freq_contrastive["balance_penalty"]
            + mean_mix[2] * joint_contrastive["balance_penalty"]
        )
        contrastive["occupancy_penalty"] = (
            mean_mix[0] * time_contrastive["occupancy_penalty"]
            + mean_mix[1] * freq_contrastive["occupancy_penalty"]
            + mean_mix[2] * joint_contrastive["occupancy_penalty"]
        )
        contrastive["assignment_ratio"] = (
            mean_mix[0] * time_contrastive["assignment_ratio"]
            + mean_mix[1] * freq_contrastive["assignment_ratio"]
            + mean_mix[2] * joint_contrastive["assignment_ratio"]
        )
        contrastive["specialization_penalty"] = (
            mean_mix[0] * time_contrastive["specialization_penalty"]
            + mean_mix[1] * freq_contrastive["specialization_penalty"]
            + mean_mix[2] * joint_contrastive["specialization_penalty"]
        )
        contrastive["specialization_gap"] = (
            mean_mix[0] * time_contrastive["specialization_gap"]
            + mean_mix[1] * freq_contrastive["specialization_gap"]
            + mean_mix[2] * joint_contrastive["specialization_gap"]
        )
        contrastive["specialization_competitor_score"] = (
            mean_mix[0] * time_contrastive["specialization_competitor_score"]
            + mean_mix[1] * freq_contrastive["specialization_competitor_score"]
            + mean_mix[2] * joint_contrastive["specialization_competitor_score"]
        )
        contrastive["prototype_diversity_penalty"] = diversity_penalty
        contrastive["prototype_diversity_scale"] = diversity_scale
        contrastive["prototype_mix_time"] = mean_mix[0]
        contrastive["prototype_mix_freq"] = mean_mix[1]
        contrastive["prototype_mix_joint"] = mean_mix[2]
        contrastive["prototype_consensus_penalty"] = consensus_penalty
        contrastive["proto_lambda_schedule"] = (
            mean_mix[0] * time_contrastive["proto_lambda_schedule"]
            + mean_mix[1] * freq_contrastive["proto_lambda_schedule"]
            + mean_mix[2] * joint_contrastive["proto_lambda_schedule"]
        )
        contrastive["specialization_lambda_schedule"] = (
            mean_mix[0] * time_contrastive["specialization_lambda_schedule"]
            + mean_mix[1] * freq_contrastive["specialization_lambda_schedule"]
            + mean_mix[2] * joint_contrastive["specialization_lambda_schedule"]
        )
        contrastive["stability_consistency_penalty"] = consistency_penalty
        contrastive["region_ensemble_std"] = region_ensemble_std
        extras["target_proto_probs"] = extras.get("joint_target_proto_probs", extras.get("target_proto_probs"))
        extras["proto_positive_scores"] = joint_contrastive["proto_positive_scores"]
        extras["proto_negative_scores"] = joint_contrastive["proto_negative_scores"]
        extras["prototype_usage"] = joint_contrastive["prototype_usage"]
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
            prototype_head=(self.prototype_head if self.config.prototype_assignment_enabled else None),
            variant_id=self.config.variant_id,
            active_components=self.active_component_summary,
        )
        payload["epoch"] = int(epoch)
        payload["stage"] = stage
        return payload
