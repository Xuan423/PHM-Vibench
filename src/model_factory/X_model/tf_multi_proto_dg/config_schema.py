from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from src.utils.utils import get_num_channels, get_num_classes


def _as_list(value: Any, default: List[str]) -> List[str]:
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


def _as_mask_mapping(
    source: Any,
    canonical_names: List[str],
    label: str,
) -> Dict[str, float]:
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
    concept_dim: int = 64
    use_contrastive_head: bool = True
    num_prototypes_per_class: int = 4
    proto_neg_k: int = 0
    proto_ema_momentum: float = 0.95
    proto_empty_reset_steps: int = 200
    readability_weight: float = 0.0
    complementarity_weight: float = 0.0
    lambda_cl_start: float = 0.0
    lambda_cl_end: float = 0.1
    temperature: float = 0.07
    export_diagnostics: bool = False
    prototype_card_top_t: int = 8
    diagnostics_max_members: int = 32
    ablation: AblationConfig = field(default_factory=AblationConfig)

    @property
    def time_operator_count(self) -> int:
        return len(self.time_operators)

    @property
    def freq_operator_count(self) -> int:
        return len(self.freq_operators)

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
        time=_coerce_unit_mask(_namespace_get(_namespace_get(raw_ablation, "branch_masks"), "time"), "model.ablation.branch_masks.time", 1.0),
        freq=_coerce_unit_mask(_namespace_get(_namespace_get(raw_ablation, "branch_masks"), "freq"), "model.ablation.branch_masks.freq", 1.0),
    )
    loss_control = LossControlConfig(
        contrastive_scale=float(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "contrastive_scale", 1.0)),
        readability_scale=float(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "readability_scale", 1.0)),
        complementarity_scale=float(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "complementarity_scale", 1.0)),
        prototype_assignment_enabled=bool(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "prototype_assignment_enabled", True)),
        prototype_update_enabled=bool(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "prototype_update_enabled", True)),
        diagnostics_enabled=bool(_namespace_get(_namespace_get(raw_ablation, "loss_control"), "diagnostics_enabled", True)),
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

    config = TFMultiProtoDGConfig(
        name=str(getattr(args_model, "name", "TF_MultiProtoDG")),
        type=str(getattr(args_model, "type", "X_model")),
        device=str(getattr(args_model, "device", "cpu")),
        input_length=input_length,
        in_channels=int(getattr(args_model, "in_channels", _infer_uniform_channels(metadata))),
        num_classes=_coerce_num_classes(metadata, getattr(args_model, "num_classes", None)),
        time_operators=time_operators,
        freq_operators=freq_operators,
        time_indicators=time_indicators,
        freq_indicators=freq_indicators,
        time_patch_count=int(getattr(args_model, "time_patch_count", 8)),
        freq_band_count=int(getattr(args_model, "freq_band_count", 8)),
        padding_mode=str(getattr(args_model, "padding_mode", "error")),
        use_topk_selector=bool(getattr(args_model, "use_topk_selector", False)),
        top_k_features=(
            int(getattr(args_model, "top_k_features"))
            if getattr(args_model, "top_k_features", None) is not None
            else None
        ),
        topk_score_mode=str(getattr(args_model, "topk_score_mode", "fisher_over_domain_var")),
        topk_ema_momentum=float(getattr(args_model, "topk_ema_momentum", 0.9)),
        topk_warmup_epochs=int(getattr(args_model, "topk_warmup_epochs", 1)),
        concept_dim=int(getattr(args_model, "concept_dim", 64)),
        use_contrastive_head=bool(getattr(args_model, "use_contrastive_head", True)),
        num_prototypes_per_class=int(getattr(args_model, "num_prototypes_per_class", 4)),
        proto_neg_k=int(getattr(args_model, "proto_neg_k", 0)),
        proto_ema_momentum=float(getattr(args_model, "proto_ema_momentum", 0.95)),
        proto_empty_reset_steps=int(getattr(args_model, "proto_empty_reset_steps", 200)),
        readability_weight=float(getattr(args_model, "readability_weight", 0.0)),
        complementarity_weight=float(getattr(args_model, "complementarity_weight", 0.0)),
        lambda_cl_start=float(getattr(args_model, "lambda_cl_start", 0.0)),
        lambda_cl_end=float(getattr(args_model, "lambda_cl_end", 0.1)),
        temperature=float(getattr(args_model, "temperature", 0.07)),
        export_diagnostics=bool(getattr(args_model, "export_diagnostics", False)),
        prototype_card_top_t=int(getattr(args_model, "prototype_card_top_t", 8)),
        diagnostics_max_members=int(getattr(args_model, "diagnostics_max_members", 32)),
        ablation=ablation,
    )

    if config.time_patch_count <= 0:
        raise ValueError("model.time_patch_count must be positive.")
    if config.freq_band_count <= 0:
        raise ValueError("model.freq_band_count must be positive.")
    if config.padding_mode not in {"error", "zero_right"}:
        raise ValueError("model.padding_mode must be either 'error' or 'zero_right'.")
    if config.concept_dim <= 0:
        raise ValueError("model.concept_dim must be positive.")
    if config.num_prototypes_per_class <= 0:
        raise ValueError("model.num_prototypes_per_class must be positive.")
    if config.proto_neg_k < 0:
        raise ValueError("model.proto_neg_k must be non-negative.")
    if not (0.0 <= config.proto_ema_momentum < 1.0):
        raise ValueError("model.proto_ema_momentum must satisfy 0 <= momentum < 1.")
    if config.prototype_card_top_t <= 0:
        raise ValueError("model.prototype_card_top_t must be positive.")
    if config.diagnostics_max_members <= 0:
        raise ValueError("model.diagnostics_max_members must be positive.")
    if config.top_k_features is not None and config.top_k_features <= 0:
        raise ValueError("model.top_k_features must be positive when provided.")
    if config.use_topk_selector and config.top_k_features is None:
        raise ValueError("model.top_k_features is required when use_topk_selector is enabled.")
    if config.ablation.enabled and not config.ablation.branch_enabled:
        raise ValueError("At least one branch mask must stay active when ablation is enabled.")
    if config.ablation.loss_control.contrastive_scale < 0.0:
        raise ValueError("model.ablation.loss_control.contrastive_scale must be non-negative.")
    if config.ablation.loss_control.readability_scale < 0.0:
        raise ValueError("model.ablation.loss_control.readability_scale must be non-negative.")
    if config.ablation.loss_control.complementarity_scale < 0.0:
        raise ValueError("model.ablation.loss_control.complementarity_scale must be non-negative.")
    if config.ablation.loss_control.prototype_update_enabled and not config.ablation.loss_control.prototype_assignment_enabled:
        raise ValueError(
            "model.ablation.loss_control.prototype_update_enabled requires prototype_assignment_enabled=true."
        )

    if config.padding_mode == "error" and config.input_length % config.time_patch_count != 0:
        raise ValueError(
            "model.input_length must be divisible by model.time_patch_count when padding_mode='error'."
        )

    freq_bins = config.input_length // 2 + 1
    if config.padding_mode == "error" and freq_bins % config.freq_band_count != 0:
        raise ValueError(
            "RFFT bin count must be divisible by model.freq_band_count when padding_mode='error'."
        )

    return config
