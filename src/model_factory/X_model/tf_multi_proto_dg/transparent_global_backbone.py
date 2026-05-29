from __future__ import annotations

from collections import OrderedDict
import math
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model_factory.X_model.utils.Feature_extract import (
    AbsMeanFeature,
    BandEnergyFeature,
    BandRMSFeature,
    ClearanceFactorFeature,
    CrestFactorFeature,
    EntropyFeature,
    KurtosisFeature,
    MeanFeature,
    PeakRatioFeature,
    RMSFeature,
    ShapeFactorFeature,
    SkewnessFeature,
    SpectralEntropyFeature,
    StdFeature,
)
from src.model_factory.X_model.utils.Signal_processing import (
    GaussianBandMask,
    HilbertTransform,
    Identity,
    IdentitySpectrum,
    LogSpectrum,
    SpectralWhitening,
    WaveFilters,
)

FEATURE_OPERATOR_REGISTRY = {
    "Mean": MeanFeature,
    "Std": StdFeature,
    "RMS": RMSFeature,
    "Kurtosis": KurtosisFeature,
    "CrestFactor": CrestFactorFeature,
    "Entropy": EntropyFeature,
    "AbsMean": AbsMeanFeature,
    "Skewness": SkewnessFeature,
    "ClearanceFactor": ClearanceFactorFeature,
    "ShapeFactor": ShapeFactorFeature,
    "BandEnergy": BandEnergyFeature,
    "BandRMS": BandRMSFeature,
    "SpectralEntropy": SpectralEntropyFeature,
    "PeakRatio": PeakRatioFeature,
}


def _get_unique_module_name(existing_names: Iterable[str], module_name: str) -> str:
    if module_name not in existing_names:
        return module_name
    index = 1
    candidate = f"{module_name}_{index}"
    existing = set(existing_names)
    while candidate in existing:
        index += 1
        candidate = f"{module_name}_{index}"
    return candidate


class _CustomBatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 0.1):
        super().__init__()
        # Keep legacy semantics from historical transparent backbone:
        # the same scalar controls running-stat momentum and denominator epsilon.
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(1, int(num_features)))
        self.register_buffer("running_var", torch.ones(1, int(num_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.eps) * self.running_mean + self.eps * mean
            self.running_var = (1 - self.eps) * self.running_var + self.eps * var
            return (x - mean) / (var.sqrt() + self.eps)
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class _LaplaceNeuralOperator(nn.Module):
    """Laplace-domain neural operator for the transparent time backbone.

    This is intentionally local to TF_MultiProtoDG's transparent backbone so LNO
    follows the same registry/forward path as HT and WF without depending on the
    standalone TSPN model wiring.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.name = "LNO"
        self.channels = max(1, int(getattr(args, "scale", 1)))
        self.modes = max(1, self.channels * self.channels)
        init_scale = 1.0 / float(max(self.channels * self.channels, 1))
        self.weights_pole = nn.Parameter(
            init_scale
            * torch.rand(
                self.channels,
                self.channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )
        self.weights_residue = nn.Parameter(
            init_scale
            * torch.rand(
                self.channels,
                self.channels,
                self.modes,
                dtype=torch.cfloat,
            )
        )

    def reset_deterministic_(self) -> None:
        idx = torch.arange(
            self.channels * self.channels * self.modes,
            device=self.weights_pole.device,
            dtype=torch.float32,
        ).view(self.channels, self.channels, self.modes)
        init_scale = 1.0 / float(max(self.channels * self.channels, 1))
        pole = torch.complex(torch.cos(idx), torch.sin(idx)) * init_scale
        residue = torch.complex(torch.sin(idx + 1.0), torch.cos(idx + 1.0)) * init_scale
        with torch.no_grad():
            self.weights_pole.copy_(pole.to(dtype=self.weights_pole.dtype))
            self.weights_residue.copy_(residue.to(dtype=self.weights_residue.dtype))

    @staticmethod
    def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps = torch.as_tensor(1e-6, device=denominator.device, dtype=denominator.real.dtype)
        safe_denominator = torch.where(
            denominator.abs() < eps,
            denominator + eps.to(dtype=denominator.dtype),
            denominator,
        )
        return numerator / safe_denominator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Current transparent backbone operators use B,L,C tensors.
        if x.ndim != 3:
            raise ValueError(f"LNO expects a B,L,C tensor, got shape={tuple(x.shape)}.")
        _, length, channels = x.shape
        if channels > self.channels:
            raise ValueError(
                f"LNO was initialized for at most {self.channels} channels, got {channels}."
            )
        if length <= 1:
            return x

        x_bcl = rearrange(x, "b l c -> b c l")
        real_dtype = x_bcl.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        alpha = torch.fft.fft(x_bcl, dim=-1)

        pole = self.weights_pole[:channels, :channels, :].to(
            device=x.device,
            dtype=complex_dtype,
        )
        residue = self.weights_residue[:channels, :channels, :].to(
            device=x.device,
            dtype=complex_dtype,
        )
        # Bound the transient exponent's real component to avoid numerical blowups
        # while keeping the learned complex pole/residue parameterization intact.
        pole_for_exp = torch.complex(
            pole.real.clamp(min=-8.0, max=8.0),
            pole.imag.clamp(min=-64.0, max=64.0),
        )

        dt = 1.0 / float(length - 1)
        lambda_freq = torch.fft.fftfreq(
            length,
            d=dt,
            device=x.device,
            dtype=real_dtype,
        )
        lambda1 = (lambda_freq * (2.0 * math.pi)).to(dtype=complex_dtype) * 1j
        lambda1 = lambda1.view(length, 1, 1, 1)
        denominator = lambda1 - pole.view(1, channels, channels, self.modes)
        transfer = self._safe_divide(
            residue.view(1, channels, channels, self.modes),
            denominator,
        )

        output_residue_freq = torch.einsum("bix,xiok->box", alpha, transfer)
        output_residue_modes = torch.einsum("bix,xiok->bok", alpha, -transfer)

        transient = torch.fft.ifft(output_residue_freq, n=length, dim=-1).real
        t = torch.linspace(0.0, 1.0, steps=length, device=x.device, dtype=real_dtype)
        response = torch.exp(
            pole_for_exp.unsqueeze(-1)
            * t.to(dtype=complex_dtype).view(1, 1, 1, length)
        )
        steady = torch.einsum("bix,ioxz->boz", output_residue_modes, response).real
        steady = steady / float(length)

        out = transient + steady
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return rearrange(out.to(dtype=x.dtype), "b c l -> b l c")


SIGNAL_OPERATOR_REGISTRY = {
    "WF": WaveFilters,
    "HT": HilbertTransform,
    "LNO": _LaplaceNeuralOperator,
    "I": Identity,
    "IdentitySpectrum": IdentitySpectrum,
    "LogSpectrum": LogSpectrum,
    "SpectralWhitening": SpectralWhitening,
    "GaussianBandMask": GaussianBandMask,
}


class _SignalProcessingLayer(nn.Module):
    def __init__(
        self,
        signal_processing_modules: Mapping[str, nn.Module],
        input_channels: int,
        output_channels: int,
        skip_connection: bool = True,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(int(input_channels))
        self.weight_connection = nn.Linear(int(input_channels), int(output_channels))
        self.signal_processing_modules = nn.ModuleDict(signal_processing_modules)
        self.module_num = int(len(signal_processing_modules))
        self.temperature = float(max(temperature, 1e-6))
        # Historical high-accuracy runs used a simplex-projected operator mixer.
        # Repeated sharpening preserves that sparse operator-route semantics without
        # mutating Parameter.data inside forward.
        self.projection_steps = 8
        self.skip_connection = (
            nn.Linear(int(input_channels), int(output_channels))
            if bool(skip_connection)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b l c -> b c l")
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, "b c l -> b l c")

        mixed_weight = self.weight_connection.weight
        for _ in range(int(self.projection_steps)):
            mixed_weight = F.softmax((1.0 / self.temperature) * mixed_weight, dim=0)
        x = F.linear(normed_x, mixed_weight, self.weight_connection.bias)

        splits = torch.split(x, x.size(2) // self.module_num, dim=2)
        outputs = []
        for module, split in zip(self.signal_processing_modules.values(), splits):
            outputs.append(module(split))
        x = torch.cat(outputs, dim=2)

        if self.skip_connection is not None:
            x = x + self.skip_connection(normed_x)
        return x


class _FeatureExtractorLayer(nn.Module):
    def __init__(
        self,
        feature_extractor_modules: Mapping[str, nn.Module],
        in_channels: int,
        out_channels: int,
        norm_mode: str = "layernorm",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.feature_extractor_modules = nn.ModuleDict(feature_extractor_modules)
        total_channels = int(len(self.feature_extractor_modules) * int(out_channels))
        self.weight_connection = nn.Linear(int(in_channels), int(out_channels))
        self.pre_norm = nn.InstanceNorm1d(int(in_channels))
        resolved_mode = str(norm_mode).lower()
        self.norm_mode = resolved_mode
        if resolved_mode == "legacy_running":
            # Keep historical behavior for ablations and backward compatibility.
            self.norm = _CustomBatchNorm(total_channels, eps=float(norm_eps))
        elif resolved_mode == "layernorm":
            # Use sample-wise normalization to reduce seed/batch-order sensitivity.
            self.norm = nn.LayerNorm(total_channels, eps=float(norm_eps))
        else:
            raise ValueError(
                f"Unsupported transparent feature norm_mode={norm_mode!r}. "
                "Expected one of {'layernorm', 'legacy_running'}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b l c -> b c l")
        x = self.pre_norm(x)
        outputs = []
        for module in self.feature_extractor_modules.values():
            outputs.append(module(x))
        features = torch.cat(outputs, dim=1).squeeze(-1)
        return self.norm(features)


class TransparentGlobalBackbone(nn.Module):
    """Independent transparent deep global backbone (time or frequency branch)."""

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.signal_processing_modules, self.feature_extractor_modules = self._config_network(args)
        self.layer_num = len(self.signal_processing_modules)
        self._init_signal_processing_layers()
        self._init_feature_extractor_layers()
        if bool(getattr(args, "deterministic_init", False)):
            self._apply_deterministic_init()

    def _config_network(self, args):
        signal_processing_modules = []
        for layer_ops in args.signal_processing_configs.values():
            layer_dict: Dict[str, nn.Module] = OrderedDict()
            for op_name in layer_ops:
                if op_name not in SIGNAL_OPERATOR_REGISTRY:
                    raise KeyError(f"Unsupported transparent module: {op_name}")
                module_name = _get_unique_module_name(layer_dict.keys(), op_name)
                layer_dict[module_name] = SIGNAL_OPERATOR_REGISTRY[op_name](args)
            signal_processing_modules.append(layer_dict)

        feature_extractor_modules: Dict[str, nn.Module] = OrderedDict()
        for feature_name in args.feature_extractor_configs:
            if feature_name not in FEATURE_OPERATOR_REGISTRY:
                raise KeyError(f"Unsupported transparent feature: {feature_name}")
            feature_extractor_modules[str(feature_name)] = FEATURE_OPERATOR_REGISTRY[feature_name]()
        return signal_processing_modules, feature_extractor_modules

    def _init_signal_processing_layers(self) -> None:
        in_channels = int(self.args.in_channels)
        out_channels = int(self.args.out_channels * self.args.scale)
        self.signal_processing_layers = nn.ModuleList()
        for layer_modules in self.signal_processing_modules:
            layer = _SignalProcessingLayer(
                signal_processing_modules=layer_modules,
                input_channels=in_channels,
                output_channels=out_channels,
                skip_connection=bool(self.args.skip_connection),
            ).to(self.args.device)
            if out_channels % max(layer.module_num, 1) != 0:
                raise ValueError(
                    f"transparent out_channels={out_channels} must be divisible "
                    f"by module_num={layer.module_num}"
                )
            self.signal_processing_layers.append(layer)
            in_channels = out_channels
        self.channel_for_feature = int(out_channels)

    def _init_feature_extractor_layers(self) -> None:
        layer = _FeatureExtractorLayer(
            feature_extractor_modules=self.feature_extractor_modules,
            in_channels=int(self.channel_for_feature),
            out_channels=int(self.channel_for_feature),
            norm_mode=str(getattr(self.args, "feature_norm_mode", "layernorm")),
            norm_eps=float(getattr(self.args, "feature_norm_eps", 1e-5)),
        ).to(self.args.device)
        self.feature_extractor_layers = layer
        self.channel_for_classifier = int(
            self.channel_for_feature * len(self.feature_extractor_modules)
        )

    @staticmethod
    def _dct_weight(
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Seed-independent smooth basis over fixed channel coordinates."""
        rows = int(rows)
        cols = int(cols)
        positions = torch.arange(cols, dtype=torch.float32, device=device).add(0.5)
        frequencies = torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1)
        basis = torch.cos(torch.pi * frequencies * positions.unsqueeze(0) / float(max(cols, 1)))
        if rows > 0:
            basis[0, :] = 1.0
        basis = basis - basis.mean(dim=1, keepdim=True)
        if rows > 0:
            basis[0, :] = 1.0
        basis = basis / basis.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return basis.to(dtype=dtype)

    @staticmethod
    def _deterministic_linear_(linear: nn.Linear, gain: float = 1.0) -> None:
        out_dim, in_dim = linear.weight.shape
        limit = math.sqrt(6.0 / float(max(out_dim + in_dim, 1))) * float(gain)
        weight = TransparentGlobalBackbone._dct_weight(
            rows=out_dim,
            cols=in_dim,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        weight = weight * limit
        with torch.no_grad():
            linear.weight.copy_(weight)
            if linear.bias is not None:
                linear.bias.zero_()

    def _apply_deterministic_init(self) -> None:
        for layer in self.signal_processing_layers:
            self._deterministic_linear_(layer.weight_connection, gain=1.0)
            if layer.skip_connection is not None:
                self._deterministic_linear_(layer.skip_connection, gain=0.5)
            for module in layer.signal_processing_modules.values():
                if hasattr(module, "reset_deterministic_"):
                    module.reset_deterministic_()
                if hasattr(module, "f_c") and hasattr(module, "f_b"):
                    channels = int(module.f_c.numel())
                    center_low = 0.05
                    center_high = 0.45
                    if channels <= 1:
                        centers = torch.tensor(
                            [0.25],
                            device=module.f_c.device,
                            dtype=module.f_c.dtype,
                        )
                    else:
                        centers = torch.linspace(
                            center_low,
                            center_high,
                            steps=channels,
                            device=module.f_c.device,
                            dtype=module.f_c.dtype,
                        )
                    bandwidth = torch.full(
                        (channels,),
                        0.08,
                        device=module.f_b.device,
                        dtype=module.f_b.dtype,
                    )
                    with torch.no_grad():
                        module.f_c.copy_(centers.view_as(module.f_c))
                        module.f_b.copy_(bandwidth.view_as(module.f_b))
        self._deterministic_linear_(self.feature_extractor_layers.weight_connection, gain=1.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.signal_processing_layers:
            hidden = layer(hidden)
        features = self.feature_extractor_layers(hidden)
        return features.view(features.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_features(x)
