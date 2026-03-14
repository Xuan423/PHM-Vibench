from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

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

    @property
    def time_operator_count(self) -> int:
        return len(self.time_operators)

    @property
    def freq_operator_count(self) -> int:
        return len(self.freq_operators)


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

    config = TFMultiProtoDGConfig(
        name=str(getattr(args_model, "name", "TF_MultiProtoDG")),
        type=str(getattr(args_model, "type", "X_model")),
        device=str(getattr(args_model, "device", "cpu")),
        input_length=input_length,
        in_channels=int(getattr(args_model, "in_channels", _infer_uniform_channels(metadata))),
        num_classes=_coerce_num_classes(metadata, getattr(args_model, "num_classes", None)),
        time_operators=_as_list(
            getattr(args_model, "time_operators", None),
            ["Identity", "FirstDifference", "DetrendMovingAverage", "HilbertEnvelope", "TKEO"],
        ),
        freq_operators=_as_list(
            getattr(args_model, "freq_operators", None),
            ["IdentitySpectrum", "LogSpectrum", "SpectralWhitening", "GaussianBandMask"],
        ),
        time_indicators=_as_list(
            getattr(args_model, "time_indicators", None),
            ["Mean", "Std", "RMS", "Kurtosis", "CrestFactor", "Entropy"],
        ),
        freq_indicators=_as_list(
            getattr(args_model, "freq_indicators", None),
            ["BandEnergy", "BandRMS", "SpectralEntropy", "PeakRatio"],
        ),
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
