from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from src.utils.utils import get_num_channels, get_num_classes


def _as_list(value: Any, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"Expected a list-like value, got {type(value)!r}.")


def _infer_uniform_channels(metadata: Any) -> int:
    channels = get_num_channels(metadata)
    if isinstance(channels, dict):
        uniq = sorted({int(v) for v in channels.values()})
        if len(uniq) != 1:
            raise ValueError(f"Inconsistent in_channels across dataset_ids: {channels}")
        return int(uniq[0])
    return int(channels)


def _coerce_num_classes(metadata: Any, provided: Any = None) -> int | Dict[str, int]:
    if provided is None:
        return get_num_classes(metadata)
    if isinstance(provided, dict):
        return {str(key): int(value) for key, value in provided.items()}
    return int(provided)


def _namespace_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coerce_positive_int(value: Any, label: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{label} must be provided.")
        value = default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if resolved <= 0:
        raise ValueError(f"{label} must be positive, got {resolved}.")
    return resolved


def _coerce_non_negative_int(value: Any, label: str, default: int = 0) -> int:
    if value is None:
        value = default
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if resolved < 0:
        raise ValueError(f"{label} must be non-negative, got {resolved}.")
    return resolved


def _coerce_float(value: Any, label: str, default: float) -> float:
    if value is None:
        value = default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _coerce_non_negative_float(value: Any, label: str, default: float) -> float:
    resolved = _coerce_float(value, label, default)
    if resolved < 0.0:
        raise ValueError(f"{label} must be non-negative, got {resolved}.")
    return resolved


def _as_mask_mapping(source: Any, canonical_names: List[str], label: str) -> Dict[str, float]:
    canonical_set = {str(name) for name in canonical_names}
    if source is None:
        return {str(name): 1.0 for name in canonical_names}
    if not isinstance(source, Mapping):
        source = vars(source)

    unexpected = sorted(str(key) for key in source.keys() if str(key) not in canonical_set)
    if unexpected:
        raise ValueError(
            f"{label} contains unknown keys: {unexpected}. Expected subset of {sorted(canonical_set)}."
        )

    resolved: Dict[str, float] = {}
    for name in canonical_names:
        value = source.get(name, 1.0)
        try:
            resolved[str(name)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{name} must be numeric.") from exc
        if resolved[str(name)] not in {0.0, 1.0}:
            raise ValueError(f"{label}.{name} must be 0.0 or 1.0, got {resolved[str(name)]}.")
    return resolved


def _coerce_unit_mask(value: Any, label: str, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    try:
        mask_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if mask_value not in {0.0, 1.0}:
        raise ValueError(f"{label} must be 0.0 or 1.0, got {mask_value}.")
    return mask_value


def _validate_choice(value: str, label: str, allowed: Iterable[str]) -> str:
    allowed_set = {str(item) for item in allowed}
    resolved = str(value)
    if resolved not in allowed_set:
        raise ValueError(f"{label} must be one of {sorted(allowed_set)}, got {resolved!r}.")
    return resolved


@dataclass(frozen=True)
class BranchMaskConfig:
    time: float = 1.0
    freq: float = 1.0


@dataclass(frozen=True)
class LossControlConfig:
    contrastive_scale: float = 1.0
    readability_scale: float = 1.0
    complementarity_scale: float = 1.0
    prototype_assignment_enabled: bool = True
    prototype_update_enabled: bool = True
    diagnostics_enabled: bool = True


@dataclass(frozen=True)
class AblationConfig:
    enabled: bool = False
    variant_id: str = "baseline"
    branch_masks: BranchMaskConfig = field(default_factory=BranchMaskConfig)
    operator_masks_time: Dict[str, float] = field(default_factory=dict)
    operator_masks_freq: Dict[str, float] = field(default_factory=dict)
    indicator_masks_time: Dict[str, float] = field(default_factory=dict)
    indicator_masks_freq: Dict[str, float] = field(default_factory=dict)
    loss_control: LossControlConfig = field(default_factory=LossControlConfig)

    @property
    def branch_enabled(self) -> bool:
        return self.branch_masks.time > 0.0 or self.branch_masks.freq > 0.0


@dataclass(frozen=True)
class ClassifierConfig:
    hypersphere_enabled: bool = False
    scale: float = 1.0
    residual_input: str = "raw_feature"
    residual_weight: float = 1.0
    residual_topk: int | None = None


@dataclass(frozen=True)
class PrototypeRoutingConfig:
    assignment_mode: str = "hard"
    assign_tau: float = 0.2
    balance_weight: float = 0.0
    update_mode: str = "learnable"


@dataclass(frozen=True)
class RegionEvidenceConfig:
    enabled: bool = False
    region_dim: int = 0
    num_slots: int = 0
    slot_tau: float = 0.3


@dataclass
class TFMultiProtoDGConfig:
    name: str
    type: str
    device: str
    input_length: int
    in_channels: int
    num_classes: int | Dict[str, int]
    time_operators: List[str] = field(default_factory=list)
    freq_operators: List[str] = field(default_factory=list)
    time_indicators: List[str] = field(default_factory=list)
    freq_indicators: List[str] = field(default_factory=list)
    time_patch_count: int = 8
    freq_band_count: int = 8
    padding_mode: str = "error"
    use_topk_selector: bool = False
    top_k_features: int | None = None
    topk_score_mode: str = "fisher_over_domain_var"
    topk_ema_momentum: float = 0.9
    topk_warmup_epochs: int = 1
    role_dim: int = 16
    concept_dim: int = 64
    use_contrastive_head: bool = True
    num_prototypes_per_class: int = 4
    prototype_temperature: float = 0.07
    prototype_class_pool_mode: str = "logsumexp"
    adaptive_class_temperature_enabled: bool = False
    adaptive_class_temperature_target_neff: float = 2.0
    adaptive_class_temperature_min_scale: float = 0.5
    adaptive_class_temperature_max_scale: float = 1.5
    adaptive_effective_k_enabled: bool = False
    adaptive_effective_k_ready_count: float = 8.0
    adaptive_effective_k_min: int = 1
    adaptive_effective_k_penalty: float = 8.0
    adaptive_contrastive_neff_enabled: bool = False
    adaptive_contrastive_neff_min_scale: float = 0.25
    adaptive_contrastive_neff_max_scale: float = 1.0
    prototype_assignment_temperature: float = 0.07
    proto_contrastive_temperature: float = 0.07
    proto_contrastive_positive_mode: str = "logsumexp"
    proto_contrastive_input: str = "h"
    proto_contrastive_use_scaled_scores: bool = False
    proto_contrastive_weight: float = 0.0
    proto_contrastive_neg_topk: int = 0
    proto_specialization_weight: float = 0.0
    proto_specialization_margin: float = 0.0
    proto_specialization_temperature: float = 0.07
    adaptive_proto_enabled: bool = False
    adaptive_proto_assignment_target: float = 0.92
    adaptive_proto_balance_target: float = 0.18
    adaptive_proto_min_scale: float = 0.0
    adaptive_proto_max_scale: float = 2.0
    adaptive_proto_balance_mix: float = 0.5
    channel_fusion_norm: str = "layernorm"
    role_nonneg: str = "none"
    role_input_norm: str = "layernorm"
    role_output_norm: str = "none"
    cross_score_norm: str = "layernorm"
    cross_pool_mode: str = "logsumexp"
    cross_pool_tau: float = 0.5
    cross_self_mix: float = 0.0
    cross_self_score_mode: str = "absmean"
    cross_adaptive_self_mix: bool = False
    concept_norm: str = "layernorm"
    prototype_logit_scale_init: float = 12.0
    prototype_diversity_weight: float = 0.0
    prototype_diversity_target_cos: float = 0.0
    prototype_diversity_adaptive_gain: float = 0.0
    prototype_occupancy_weight: float = 0.0
    export_diagnostics: bool = False
    prototype_card_top_t: int = 8
    diagnostics_max_members: int = 32
    proto_neg_k: int = 0
    proto_ema_momentum: float = 0.95
    proto_empty_reset_steps: int = 200
    readability_weight: float = 0.0
    complementarity_weight: float = 0.0
    lambda_cl_start: float = 0.0
    lambda_cl_end: float = 0.0
    contrastive_warmup_epochs: int = 0
    temperature: float = 0.07
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    prototype: PrototypeRoutingConfig = field(default_factory=PrototypeRoutingConfig)
    region_evidence: RegionEvidenceConfig = field(default_factory=RegionEvidenceConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @property
    def time_operator_count(self) -> int:
        return len(self.time_operators)

    @property
    def freq_operator_count(self) -> int:
        return len(self.freq_operators)

    @property
    def concept_output_dim(self) -> int:
        return int(self.concept_dim)

    @property
    def prototype_assignment_enabled(self) -> bool:
        return bool(self.ablation.loss_control.prototype_assignment_enabled)

    @property
    def prototype_update_enabled(self) -> bool:
        return bool(self.ablation.loss_control.prototype_update_enabled)

    @property
    def diagnostics_enabled(self) -> bool:
        return bool(self.export_diagnostics and self.ablation.loss_control.diagnostics_enabled)

    @property
    def contrastive_scale(self) -> float:
        return float(self.ablation.loss_control.contrastive_scale)

    @property
    def variant_id(self) -> str:
        return str(self.ablation.variant_id)


def _resolve_proto_weight(args_model: Any, lambda_cl_start: float, lambda_cl_end: float) -> float:
    if hasattr(args_model, "proto_contrastive_weight"):
        return _coerce_non_negative_float(
            getattr(args_model, "proto_contrastive_weight"),
            "model.proto_contrastive_weight",
            0.0,
        )
    return max(float(lambda_cl_start), float(lambda_cl_end))


def _min_class_count(num_classes: int | Dict[str, int]) -> int:
    if isinstance(num_classes, dict):
        return min(int(value) for value in num_classes.values())
    return int(num_classes)


def build_model_config(args_model: Any, metadata: Any) -> TFMultiProtoDGConfig:
    input_length = int(
        getattr(
            args_model,
            "input_length",
            getattr(args_model, "in_dim", getattr(args_model, "window_size", 0)),
        )
    )
    if input_length <= 1:
        raise ValueError("TF_MultiProtoDG requires a positive model.input_length or model.in_dim.")

    time_operators = _as_list(
        getattr(args_model, "time_operators", None),
        ["Identity", "FirstDifference", "DetrendMovingAverage", "HilbertEnvelope", "TKEO"],
    )
    freq_operators = _as_list(
        getattr(args_model, "freq_operators", None),
        ["IdentitySpectrum", "LogSpectrum", "SpectralWhitening", "GaussianBandMask"],
    )
    time_indicators = _as_list(
        getattr(args_model, "time_indicators", None),
        ["Mean", "Std", "RMS", "Kurtosis", "CrestFactor", "Entropy"],
    )
    freq_indicators = _as_list(
        getattr(args_model, "freq_indicators", None),
        ["BandEnergy", "BandRMS", "SpectralEntropy", "PeakRatio"],
    )

    raw_ablation = getattr(args_model, "ablation", None)
    branch_masks = BranchMaskConfig(
        time=_coerce_unit_mask(
            _namespace_get(_namespace_get(raw_ablation, "branch_masks"), "time"),
            "model.ablation.branch_masks.time",
            1.0,
        ),
        freq=_coerce_unit_mask(
            _namespace_get(_namespace_get(raw_ablation, "branch_masks"), "freq"),
            "model.ablation.branch_masks.freq",
            1.0,
        ),
    )
    loss_control = LossControlConfig(
        contrastive_scale=float(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "contrastive_scale", 1.0)
        ),
        readability_scale=float(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "readability_scale", 1.0)
        ),
        complementarity_scale=float(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "complementarity_scale", 1.0)
        ),
        prototype_assignment_enabled=bool(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "prototype_assignment_enabled", True)
        ),
        prototype_update_enabled=bool(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "prototype_update_enabled", True)
        ),
        diagnostics_enabled=bool(
            _namespace_get(_namespace_get(raw_ablation, "loss_control"), "diagnostics_enabled", True)
        ),
    )
    ablation = AblationConfig(
        enabled=bool(_namespace_get(raw_ablation, "enabled", False)),
        variant_id=str(_namespace_get(raw_ablation, "variant_id", "baseline")),
        branch_masks=branch_masks,
        operator_masks_time=_as_mask_mapping(
            _namespace_get(_namespace_get(raw_ablation, "operator_masks"), "time"),
            time_operators,
            "model.ablation.operator_masks.time",
        ),
        operator_masks_freq=_as_mask_mapping(
            _namespace_get(_namespace_get(raw_ablation, "operator_masks"), "freq"),
            freq_operators,
            "model.ablation.operator_masks.freq",
        ),
        indicator_masks_time=_as_mask_mapping(
            _namespace_get(_namespace_get(raw_ablation, "indicator_masks"), "time"),
            time_indicators,
            "model.ablation.indicator_masks.time",
        ),
        indicator_masks_freq=_as_mask_mapping(
            _namespace_get(_namespace_get(raw_ablation, "indicator_masks"), "freq"),
            freq_indicators,
            "model.ablation.indicator_masks.freq",
        ),
        loss_control=loss_control,
    )

    lambda_cl_start = _coerce_non_negative_float(
        getattr(args_model, "lambda_cl_start", 0.0),
        "model.lambda_cl_start",
        0.0,
    )
    lambda_cl_end = _coerce_non_negative_float(
        getattr(args_model, "lambda_cl_end", 0.0),
        "model.lambda_cl_end",
        0.0,
    )
    contrastive_warmup_epochs = _coerce_non_negative_int(
        getattr(args_model, "contrastive_warmup_epochs", 0),
        "model.contrastive_warmup_epochs",
        0,
    )
    concept_dim_arg = getattr(args_model, "concept_dim", None)
    role_dim = _coerce_positive_int(
        getattr(args_model, "role_dim", None),
        "model.role_dim",
        default=max(1, int(concept_dim_arg or 64) // 4),
    )
    concept_dim = int(concept_dim_arg) if concept_dim_arg is not None else 4 * role_dim
    if concept_dim != 4 * role_dim:
        raise ValueError(
            f"model.concept_dim must equal 4 * model.role_dim for the simplified path, got "
            f"concept_dim={concept_dim}, role_dim={role_dim}."
        )

    prototype_temperature = _coerce_float(
        getattr(args_model, "prototype_temperature", getattr(args_model, "temperature", 0.07)),
        "model.prototype_temperature",
        0.07,
    )
    if prototype_temperature <= 0.0:
        raise ValueError(
            f"model.prototype_temperature must be positive, got {prototype_temperature}."
        )
    prototype_class_pool_mode = _validate_choice(
        str(getattr(args_model, "prototype_class_pool_mode", "logsumexp")),
        "model.prototype_class_pool_mode",
        {"logsumexp", "max"},
    )
    adaptive_class_temperature_enabled = bool(
        getattr(args_model, "adaptive_class_temperature_enabled", False)
    )
    adaptive_class_temperature_target_neff = _coerce_non_negative_float(
        getattr(args_model, "adaptive_class_temperature_target_neff", 2.0),
        "model.adaptive_class_temperature_target_neff",
        2.0,
    )
    adaptive_class_temperature_min_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_class_temperature_min_scale", 0.5),
        "model.adaptive_class_temperature_min_scale",
        0.5,
    )
    adaptive_class_temperature_max_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_class_temperature_max_scale", 1.5),
        "model.adaptive_class_temperature_max_scale",
        1.5,
    )
    if adaptive_class_temperature_target_neff <= 0.0:
        raise ValueError(
            "model.adaptive_class_temperature_target_neff must be positive, "
            f"got {adaptive_class_temperature_target_neff}."
        )
    if adaptive_class_temperature_min_scale > adaptive_class_temperature_max_scale:
        raise ValueError(
            "model.adaptive_class_temperature_min_scale must be <= "
            "model.adaptive_class_temperature_max_scale."
        )
    adaptive_effective_k_enabled = bool(getattr(args_model, "adaptive_effective_k_enabled", False))
    adaptive_effective_k_ready_count = _coerce_non_negative_float(
        getattr(args_model, "adaptive_effective_k_ready_count", 8.0),
        "model.adaptive_effective_k_ready_count",
        8.0,
    )
    adaptive_effective_k_min = _coerce_positive_int(
        getattr(args_model, "adaptive_effective_k_min", 1),
        "model.adaptive_effective_k_min",
        1,
    )
    adaptive_effective_k_penalty = _coerce_non_negative_float(
        getattr(args_model, "adaptive_effective_k_penalty", 8.0),
        "model.adaptive_effective_k_penalty",
        8.0,
    )
    adaptive_contrastive_neff_enabled = bool(
        getattr(args_model, "adaptive_contrastive_neff_enabled", False)
    )
    adaptive_contrastive_neff_min_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_contrastive_neff_min_scale", 0.25),
        "model.adaptive_contrastive_neff_min_scale",
        0.25,
    )
    adaptive_contrastive_neff_max_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_contrastive_neff_max_scale", 1.0),
        "model.adaptive_contrastive_neff_max_scale",
        1.0,
    )
    if adaptive_contrastive_neff_min_scale > adaptive_contrastive_neff_max_scale:
        raise ValueError(
            "model.adaptive_contrastive_neff_min_scale must be <= "
            "model.adaptive_contrastive_neff_max_scale."
        )
    prototype_assignment_temperature = _coerce_float(
        getattr(args_model, "prototype_assignment_temperature", prototype_temperature),
        "model.prototype_assignment_temperature",
        prototype_temperature,
    )
    if prototype_assignment_temperature <= 0.0:
        raise ValueError(
            "model.prototype_assignment_temperature must be positive, "
            f"got {prototype_assignment_temperature}."
        )
    proto_contrastive_temperature = _coerce_float(
        getattr(args_model, "proto_contrastive_temperature", prototype_temperature),
        "model.proto_contrastive_temperature",
        prototype_temperature,
    )
    if proto_contrastive_temperature <= 0.0:
        raise ValueError(
            "model.proto_contrastive_temperature must be positive, "
            f"got {proto_contrastive_temperature}."
        )
    proto_contrastive_positive_mode = _validate_choice(
        str(getattr(args_model, "proto_contrastive_positive_mode", "logsumexp")),
        "model.proto_contrastive_positive_mode",
        {"logsumexp", "softmax_expectation", "assigned"},
    )
    proto_contrastive_input = _validate_choice(
        str(getattr(args_model, "proto_contrastive_input", "h")),
        "model.proto_contrastive_input",
        {"h", "h_core"},
    )
    proto_contrastive_use_scaled_scores = bool(
        getattr(args_model, "proto_contrastive_use_scaled_scores", False)
    )
    proto_contrastive_weight = _resolve_proto_weight(args_model, lambda_cl_start, lambda_cl_end)
    proto_contrastive_neg_topk = _coerce_non_negative_int(
        getattr(args_model, "proto_contrastive_neg_topk", getattr(args_model, "proto_neg_k", 0)),
        "model.proto_contrastive_neg_topk",
        0,
    )
    proto_specialization_weight = _coerce_non_negative_float(
        getattr(args_model, "proto_specialization_weight", 0.0),
        "model.proto_specialization_weight",
        0.0,
    )
    proto_specialization_margin = _coerce_non_negative_float(
        getattr(args_model, "proto_specialization_margin", 0.0),
        "model.proto_specialization_margin",
        0.0,
    )
    proto_specialization_temperature = _coerce_float(
        getattr(args_model, "proto_specialization_temperature", prototype_assignment_temperature),
        "model.proto_specialization_temperature",
        prototype_assignment_temperature,
    )
    if proto_specialization_temperature <= 0.0:
        raise ValueError(
            "model.proto_specialization_temperature must be positive, "
            f"got {proto_specialization_temperature}."
        )
    adaptive_proto_enabled = bool(getattr(args_model, "adaptive_proto_enabled", False))
    adaptive_proto_assignment_target = _coerce_non_negative_float(
        getattr(args_model, "adaptive_proto_assignment_target", 0.92),
        "model.adaptive_proto_assignment_target",
        0.92,
    )
    adaptive_proto_balance_target = _coerce_non_negative_float(
        getattr(args_model, "adaptive_proto_balance_target", 0.18),
        "model.adaptive_proto_balance_target",
        0.18,
    )
    adaptive_proto_min_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_proto_min_scale", 0.0),
        "model.adaptive_proto_min_scale",
        0.0,
    )
    adaptive_proto_max_scale = _coerce_non_negative_float(
        getattr(args_model, "adaptive_proto_max_scale", 2.0),
        "model.adaptive_proto_max_scale",
        2.0,
    )
    adaptive_proto_balance_mix = _coerce_non_negative_float(
        getattr(args_model, "adaptive_proto_balance_mix", 0.5),
        "model.adaptive_proto_balance_mix",
        0.5,
    )
    if adaptive_proto_assignment_target > 1.0:
        raise ValueError(
            f"model.adaptive_proto_assignment_target must be in [0, 1], got {adaptive_proto_assignment_target}."
        )
    if adaptive_proto_min_scale > adaptive_proto_max_scale:
        raise ValueError("model.adaptive_proto_min_scale must be <= model.adaptive_proto_max_scale.")

    classifier_raw = getattr(args_model, "classifier", None)
    classifier_residual_input = _validate_choice(
        str(_namespace_get(classifier_raw, "residual_input", "raw_feature")),
        "model.classifier.residual_input",
        {"raw_feature", "structured_masked", "structured_topk"},
    )
    classifier_residual_topk_raw = _namespace_get(classifier_raw, "residual_topk", None)
    classifier_residual_topk = (
        _coerce_positive_int(
            classifier_residual_topk_raw,
            "model.classifier.residual_topk",
        )
        if classifier_residual_topk_raw is not None
        else None
    )
    classifier = ClassifierConfig(
        hypersphere_enabled=bool(_namespace_get(classifier_raw, "hypersphere_enabled", False)),
        scale=_coerce_float(_namespace_get(classifier_raw, "scale", 1.0), "model.classifier.scale", 1.0),
        residual_input=classifier_residual_input,
        residual_weight=_coerce_float(
            _namespace_get(classifier_raw, "residual_weight", 1.0),
            "model.classifier.residual_weight",
            1.0,
        ),
        residual_topk=classifier_residual_topk,
    )

    prototype_raw = getattr(args_model, "prototype", None)
    assignment_mode = _validate_choice(
        str(
            _namespace_get(
                prototype_raw,
                "assignment_mode",
                getattr(args_model, "prototype_assignment_mode", "hard"),
            )
        ),
        "model.prototype.assignment_mode",
        {"hard", "soft_similarity", "soft_balanced"},
    )
    prototype = PrototypeRoutingConfig(
        assignment_mode=assignment_mode,
        assign_tau=_coerce_float(
            _namespace_get(prototype_raw, "assign_tau", getattr(args_model, "prototype_assign_tau", 0.2)),
            "model.prototype.assign_tau",
            0.2,
        ),
        balance_weight=_coerce_non_negative_float(
            _namespace_get(prototype_raw, "balance_weight", 0.0),
            "model.prototype.balance_weight",
            0.0,
        ),
        update_mode=str(_namespace_get(prototype_raw, "update_mode", "learnable")),
    )

    region_raw = getattr(args_model, "region_evidence", None)
    region_evidence = RegionEvidenceConfig(
        enabled=bool(_namespace_get(region_raw, "enabled", False)),
        region_dim=_coerce_non_negative_int(
            _namespace_get(region_raw, "region_dim", 0),
            "model.region_evidence.region_dim",
            0,
        ),
        num_slots=_coerce_non_negative_int(
            _namespace_get(region_raw, "num_slots", 0),
            "model.region_evidence.num_slots",
            0,
        ),
        slot_tau=_coerce_float(
            _namespace_get(region_raw, "slot_tau", 0.3),
            "model.region_evidence.slot_tau",
            0.3,
        ),
    )
    if region_evidence.enabled and region_evidence.num_slots <= 0:
        raise ValueError("model.region_evidence.num_slots must be positive when region evidence is enabled.")

    num_classes = _coerce_num_classes(metadata, getattr(args_model, "num_classes", None))
    num_prototypes_per_class = _coerce_positive_int(
        getattr(args_model, "num_prototypes_per_class", 4),
        "model.num_prototypes_per_class",
        4,
    )
    if adaptive_effective_k_min > num_prototypes_per_class:
        raise ValueError(
            "model.adaptive_effective_k_min must be <= model.num_prototypes_per_class."
        )
    min_classes = _min_class_count(num_classes)
    max_negatives = max(0, (min_classes - 1) * num_prototypes_per_class)
    if proto_contrastive_neg_topk > max_negatives and max_negatives > 0:
        raise ValueError(
            f"model.proto_contrastive_neg_topk={proto_contrastive_neg_topk} exceeds the maximum "
            f"available negatives {max_negatives} for the selected class counts."
        )
    if proto_contrastive_neg_topk > 0 and max_negatives == 0:
        raise ValueError("model.proto_contrastive_neg_topk requires at least two classes.")

    channel_fusion_norm = _validate_choice(
        str(getattr(args_model, "channel_fusion_norm", "layernorm")),
        "model.channel_fusion_norm",
        {"layernorm", "l2", "none"},
    )
    role_nonneg = _validate_choice(
        str(getattr(args_model, "role_nonneg", "none")),
        "model.role_nonneg",
        {"softplus", "relu", "abs", "none"},
    )
    role_input_norm = _validate_choice(
        str(getattr(args_model, "role_input_norm", "layernorm")),
        "model.role_input_norm",
        {"layernorm", "l2", "none"},
    )
    role_output_norm = _validate_choice(
        str(getattr(args_model, "role_output_norm", "none")),
        "model.role_output_norm",
        {"layernorm", "l2", "none"},
    )
    cross_score_norm = _validate_choice(
        str(getattr(args_model, "cross_score_norm", "layernorm")),
        "model.cross_score_norm",
        {"layernorm", "l2", "none"},
    )
    cross_pool_mode = _validate_choice(
        str(getattr(args_model, "cross_pool_mode", "logsumexp")),
        "model.cross_pool_mode",
        {"max", "mean", "logsumexp"},
    )
    cross_pool_tau = _coerce_float(
        getattr(args_model, "cross_pool_tau", 0.5),
        "model.cross_pool_tau",
        0.5,
    )
    if cross_pool_tau <= 0.0:
        raise ValueError(f"model.cross_pool_tau must be positive, got {cross_pool_tau}.")
    cross_self_mix = _coerce_float(
        getattr(args_model, "cross_self_mix", 0.0),
        "model.cross_self_mix",
        0.0,
    )
    if not 0.0 <= cross_self_mix <= 1.0:
        raise ValueError(f"model.cross_self_mix must be in [0, 1], got {cross_self_mix}.")
    cross_self_score_mode = _validate_choice(
        str(getattr(args_model, "cross_self_score_mode", "absmean")),
        "model.cross_self_score_mode",
        {"absmean", "l2", "none"},
    )
    cross_adaptive_self_mix = bool(getattr(args_model, "cross_adaptive_self_mix", False))
    concept_norm = _validate_choice(
        str(getattr(args_model, "concept_norm", "layernorm")),
        "model.concept_norm",
        {"layernorm", "l2", "none"},
    )
    prototype_logit_scale_init = _coerce_float(
        getattr(args_model, "prototype_logit_scale_init", 12.0),
        "model.prototype_logit_scale_init",
        12.0,
    )
    if prototype_logit_scale_init <= 0.0:
        raise ValueError(
            f"model.prototype_logit_scale_init must be positive, got {prototype_logit_scale_init}."
        )
    prototype_diversity_weight = _coerce_non_negative_float(
        getattr(args_model, "prototype_diversity_weight", 0.0),
        "model.prototype_diversity_weight",
        0.0,
    )
    prototype_diversity_target_cos = _coerce_float(
        getattr(args_model, "prototype_diversity_target_cos", 0.0),
        "model.prototype_diversity_target_cos",
        0.0,
    )
    if not -1.0 <= prototype_diversity_target_cos <= 1.0:
        raise ValueError(
            "model.prototype_diversity_target_cos must be in [-1, 1], "
            f"got {prototype_diversity_target_cos}."
        )
    prototype_diversity_adaptive_gain = _coerce_non_negative_float(
        getattr(args_model, "prototype_diversity_adaptive_gain", 0.0),
        "model.prototype_diversity_adaptive_gain",
        0.0,
    )
    prototype_occupancy_weight = _coerce_non_negative_float(
        getattr(args_model, "prototype_occupancy_weight", 0.0),
        "model.prototype_occupancy_weight",
        0.0,
    )

    use_contrastive_head = bool(
        getattr(
            args_model,
            "use_contrastive_head",
            (
                proto_contrastive_weight > 0.0
                or proto_specialization_weight > 0.0
                or lambda_cl_end > 0.0
                or lambda_cl_start > 0.0
            ),
        )
    )

    config = TFMultiProtoDGConfig(
        name=str(getattr(args_model, "name", "TF_MultiProtoDG")),
        type=str(getattr(args_model, "type", "X_model")),
        device=str(getattr(args_model, "device", "cpu")),
        input_length=input_length,
        in_channels=int(getattr(args_model, "in_channels", _infer_uniform_channels(metadata))),
        num_classes=num_classes,
        time_operators=time_operators,
        freq_operators=freq_operators,
        time_indicators=time_indicators,
        freq_indicators=freq_indicators,
        time_patch_count=_coerce_positive_int(
            getattr(args_model, "time_patch_count", 8),
            "model.time_patch_count",
            8,
        ),
        freq_band_count=_coerce_positive_int(
            getattr(args_model, "freq_band_count", 8),
            "model.freq_band_count",
            8,
        ),
        padding_mode=str(getattr(args_model, "padding_mode", "error")),
        use_topk_selector=bool(getattr(args_model, "use_topk_selector", False)),
        top_k_features=getattr(args_model, "top_k_features", None),
        topk_score_mode=str(getattr(args_model, "topk_score_mode", "fisher_over_domain_var")),
        topk_ema_momentum=float(getattr(args_model, "topk_ema_momentum", 0.9)),
        topk_warmup_epochs=int(getattr(args_model, "topk_warmup_epochs", 1)),
        role_dim=role_dim,
        concept_dim=concept_dim,
        use_contrastive_head=use_contrastive_head,
        num_prototypes_per_class=num_prototypes_per_class,
        prototype_temperature=prototype_temperature,
        prototype_class_pool_mode=prototype_class_pool_mode,
        adaptive_class_temperature_enabled=adaptive_class_temperature_enabled,
        adaptive_class_temperature_target_neff=adaptive_class_temperature_target_neff,
        adaptive_class_temperature_min_scale=adaptive_class_temperature_min_scale,
        adaptive_class_temperature_max_scale=adaptive_class_temperature_max_scale,
        adaptive_effective_k_enabled=adaptive_effective_k_enabled,
        adaptive_effective_k_ready_count=adaptive_effective_k_ready_count,
        adaptive_effective_k_min=adaptive_effective_k_min,
        adaptive_effective_k_penalty=adaptive_effective_k_penalty,
        adaptive_contrastive_neff_enabled=adaptive_contrastive_neff_enabled,
        adaptive_contrastive_neff_min_scale=adaptive_contrastive_neff_min_scale,
        adaptive_contrastive_neff_max_scale=adaptive_contrastive_neff_max_scale,
        prototype_assignment_temperature=prototype_assignment_temperature,
        proto_contrastive_temperature=proto_contrastive_temperature,
        proto_contrastive_positive_mode=proto_contrastive_positive_mode,
        proto_contrastive_input=proto_contrastive_input,
        proto_contrastive_use_scaled_scores=proto_contrastive_use_scaled_scores,
        proto_contrastive_weight=proto_contrastive_weight,
        proto_contrastive_neg_topk=proto_contrastive_neg_topk,
        proto_specialization_weight=proto_specialization_weight,
        proto_specialization_margin=proto_specialization_margin,
        proto_specialization_temperature=proto_specialization_temperature,
        adaptive_proto_enabled=adaptive_proto_enabled,
        adaptive_proto_assignment_target=adaptive_proto_assignment_target,
        adaptive_proto_balance_target=adaptive_proto_balance_target,
        adaptive_proto_min_scale=adaptive_proto_min_scale,
        adaptive_proto_max_scale=adaptive_proto_max_scale,
        adaptive_proto_balance_mix=adaptive_proto_balance_mix,
        channel_fusion_norm=channel_fusion_norm,
        role_nonneg=role_nonneg,
        role_input_norm=role_input_norm,
        role_output_norm=role_output_norm,
        cross_score_norm=cross_score_norm,
        cross_pool_mode=cross_pool_mode,
        cross_pool_tau=cross_pool_tau,
        cross_self_mix=cross_self_mix,
        cross_self_score_mode=cross_self_score_mode,
        cross_adaptive_self_mix=cross_adaptive_self_mix,
        concept_norm=concept_norm,
        prototype_logit_scale_init=prototype_logit_scale_init,
        prototype_diversity_weight=prototype_diversity_weight,
        prototype_diversity_target_cos=prototype_diversity_target_cos,
        prototype_diversity_adaptive_gain=prototype_diversity_adaptive_gain,
        prototype_occupancy_weight=prototype_occupancy_weight,
        export_diagnostics=bool(getattr(args_model, "export_diagnostics", False)),
        prototype_card_top_t=int(getattr(args_model, "prototype_card_top_t", 8)),
        diagnostics_max_members=int(getattr(args_model, "diagnostics_max_members", 32)),
        proto_neg_k=proto_contrastive_neg_topk,
        proto_ema_momentum=float(getattr(args_model, "proto_ema_momentum", 0.95)),
        proto_empty_reset_steps=int(getattr(args_model, "proto_empty_reset_steps", 200)),
        readability_weight=float(getattr(args_model, "readability_weight", 0.0)),
        complementarity_weight=float(getattr(args_model, "complementarity_weight", 0.0)),
        lambda_cl_start=lambda_cl_start,
        lambda_cl_end=lambda_cl_end,
        contrastive_warmup_epochs=contrastive_warmup_epochs,
        temperature=prototype_temperature,
        classifier=classifier,
        prototype=prototype,
        region_evidence=region_evidence,
        ablation=ablation,
    )
    if config.top_k_features is not None and int(config.top_k_features) <= 0:
        raise ValueError("model.top_k_features must be positive when provided.")
    return config
