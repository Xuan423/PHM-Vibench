from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_ops import entropy_from_abs, normalize_time, to_bcl


def _split_sizes(total: int, parts: int) -> list[int]:
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}.")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


class _IdentityOp(nn.Module):
    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        return x_bcl


class _HilbertEnvelopeOp(nn.Module):
    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        length = x_bcl.shape[-1]
        spectrum = torch.fft.fft(x_bcl, dim=-1)
        h = torch.zeros(length, device=x_bcl.device, dtype=spectrum.dtype)
        if length % 2 == 0:
            h[0] = 1
            h[length // 2] = 1
            h[1:length // 2] = 2
        else:
            h[0] = 1
            h[1:(length + 1) // 2] = 2
        analytic = torch.fft.ifft(spectrum * h, dim=-1)
        return normalize_time(analytic.abs().real)


class _WaveFilterOp(nn.Module):
    def __init__(self, kernel_size: int = 9) -> None:
        super().__init__()
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.detail_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        smooth = F.avg_pool1d(
            F.pad(x_bcl, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        )
        detail = x_bcl - smooth
        gain = torch.tanh(self.detail_gain)
        return normalize_time(smooth + gain * detail)


class _LogCompressOp(nn.Module):
    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        return normalize_time(torch.sign(x_bcl) * torch.log1p(x_bcl.abs()))


class _FirstDiffOp(nn.Module):
    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        diff = x_bcl[..., 1:] - x_bcl[..., :-1]
        return normalize_time(F.pad(diff, (1, 0)))


class _GaussianBandMaskOp(nn.Module):
    def __init__(self, init_center: float = 0.5, init_sigma: float = 0.15) -> None:
        super().__init__()
        self.center_logit = nn.Parameter(torch.tensor(float(init_center)).logit())
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(init_sigma)).clamp_min(1e-3)))

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        length = int(x_bcl.shape[-1])
        grid = torch.linspace(0.0, 1.0, length, device=x_bcl.device, dtype=x_bcl.dtype)
        center = torch.sigmoid(self.center_logit)
        sigma = self.log_sigma.exp().clamp_min(1e-3)
        mask = torch.exp(-0.5 * ((grid - center) / sigma) ** 2)
        mask = mask / mask.norm(p=2).clamp_min(1e-6)
        return normalize_time(x_bcl * mask.view(1, 1, -1))


def _canonical_module_name(name: str) -> str:
    table = {
        "I": "identity",
        "Identity": "identity",
        "IdentitySpectrum": "identity",
        "HT": "hilbert",
        "HilbertEnvelope": "hilbert",
        "WF": "wavefilter",
        "DetrendMovingAverage": "wavefilter",
        "SpectralWhitening": "wavefilter",
        "Log": "log",
        "LogSpectrum": "log",
        "FirstDifference": "diff",
        "GaussianBandMask": "gaussian",
    }
    key = str(name)
    if key not in table:
        raise ValueError(
            f"Unsupported transparent_backbone module {name!r}. "
            f"Supported keys: {sorted(table.keys())}."
        )
    return table[key]


def _build_module(name: str) -> nn.Module:
    canonical = _canonical_module_name(name)
    if canonical == "identity":
        return _IdentityOp()
    if canonical == "hilbert":
        return _HilbertEnvelopeOp()
    if canonical == "wavefilter":
        return _WaveFilterOp()
    if canonical == "log":
        return _LogCompressOp()
    if canonical == "diff":
        return _FirstDiffOp()
    if canonical == "gaussian":
        return _GaussianBandMaskOp()
    raise ValueError(f"Unsupported transparent module canonical key: {canonical}.")


class TransparentSignalLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        module_names: Sequence[str],
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")

        unique_module_names = [str(name) for name in module_names]
        if not unique_module_names:
            unique_module_names = ["I"]
        if len(unique_module_names) > out_channels:
            unique_module_names = unique_module_names[:out_channels]

        self.module_names = unique_module_names
        self.norm = nn.InstanceNorm1d(in_channels)
        self.weight_connection = nn.Linear(in_channels, out_channels)
        self.channel_splits = _split_sizes(out_channels, len(self.module_names))
        self.signal_ops = nn.ModuleList([_build_module(name) for name in self.module_names])
        self.skip_connection = nn.Linear(in_channels, out_channels) if skip_connection else None
        self.temperature = 0.1

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        x_bcl = x_blc.permute(0, 2, 1)
        normed_bcl = self.norm(x_bcl)
        normed_blc = normed_bcl.permute(0, 2, 1)
        # Match original TSPN channel-mixing semantics to preserve deep expressivity.
        soft_weight = F.softmax((1.0 / self.temperature) * self.weight_connection.weight, dim=0)
        mixed_blc = F.linear(normed_blc, soft_weight, self.weight_connection.bias)
        mixed_bcl = mixed_blc.permute(0, 2, 1)

        outputs: list[torch.Tensor] = []
        start = 0
        for channel_count, operator in zip(self.channel_splits, self.signal_ops):
            end = start + channel_count
            outputs.append(operator(mixed_bcl[:, start:end, :]))
            start = end
        out_bcl = torch.cat(outputs, dim=1)

        if self.skip_connection is not None:
            skip_blc = self.skip_connection(normed_blc)
            out_bcl = out_bcl + skip_blc.permute(0, 2, 1)
        out_bcl = torch.nan_to_num(out_bcl, nan=0.0, posinf=1e4, neginf=-1e4)
        return out_bcl.permute(0, 2, 1)


class _FeatureBatchNorm(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-4) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(1, self.num_features))
        self.register_buffer("running_var", torch.ones(1, self.num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var.detach()
            return (x - mean) / (var.sqrt() + self.eps)
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class TransparentFeatureExtractor(nn.Module):
    def __init__(self, feature_names: Iterable[str], channels: int) -> None:
        super().__init__()
        self.feature_names = [str(name) for name in feature_names]
        if not self.feature_names:
            raise ValueError("transparent_backbone_features cannot be empty.")
        self.pre_norm = nn.InstanceNorm1d(int(channels))
        self.norm = _FeatureBatchNorm(int(channels) * len(self.feature_names))

    @staticmethod
    def _feature(name: str, x_bcl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if name == "Mean":
            return x_bcl.mean(dim=-1)
        if name == "Std":
            return x_bcl.std(dim=-1, unbiased=False)
        if name == "Var":
            return x_bcl.var(dim=-1, unbiased=False)
        if name == "Entropy":
            return entropy_from_abs(x_bcl, eps=eps)
        if name == "Max":
            return x_bcl.amax(dim=-1)
        if name == "Min":
            return x_bcl.amin(dim=-1)
        if name == "AbsMean":
            return x_bcl.abs().mean(dim=-1)
        if name == "Kurtosis":
            centered = x_bcl - x_bcl.mean(dim=-1, keepdim=True)
            std = centered.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
            return (centered / std).pow(4).mean(dim=-1)
        if name == "RMS":
            return torch.sqrt(x_bcl.square().mean(dim=-1) + eps)
        if name == "CrestFactor":
            peak = x_bcl.abs().amax(dim=-1)
            rms = torch.sqrt(x_bcl.square().mean(dim=-1) + eps)
            return peak / rms.clamp_min(eps)
        if name == "Skewness":
            centered = x_bcl - x_bcl.mean(dim=-1, keepdim=True)
            std = centered.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
            return (centered / std).pow(3).mean(dim=-1)
        if name == "ClearanceFactor":
            peak = x_bcl.abs().amax(dim=-1)
            root_abs_mean = torch.sqrt(x_bcl.abs().mean(dim=-1).clamp_min(eps))
            return peak / root_abs_mean.square().clamp_min(eps)
        if name == "ShapeFactor":
            rms = torch.sqrt(x_bcl.square().mean(dim=-1) + eps)
            abs_mean = x_bcl.abs().mean(dim=-1).clamp_min(eps)
            return rms / abs_mean
        if name == "BandEnergy":
            return x_bcl.square().sum(dim=-1)
        if name == "BandRMS":
            return torch.sqrt(x_bcl.square().mean(dim=-1) + eps)
        if name == "SpectralEntropy":
            return entropy_from_abs(x_bcl, eps=eps)
        if name == "PeakRatio":
            return x_bcl.amax(dim=-1) / x_bcl.mean(dim=-1).clamp_min(eps)
        raise ValueError(
            f"Unsupported transparent_backbone feature {name!r}. "
            "Supported keys: Mean/Std/Var/Entropy/Max/Min/AbsMean/Kurtosis/RMS/"
            "CrestFactor/Skewness/ClearanceFactor/ShapeFactor/BandEnergy/BandRMS/"
            "SpectralEntropy/PeakRatio."
        )

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        x_bcl = x_blc.permute(0, 2, 1)
        x_bcl = self.pre_norm(x_bcl)
        values = [self._feature(name, x_bcl) for name in self.feature_names]
        stacked = torch.cat(values, dim=1)
        stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
        return self.norm(stacked)


class TransparentBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers: int,
        modules: Sequence[str],
        features: Sequence[str],
        out_channels: int,
        scale: int,
        skip_connection: bool,
    ) -> None:
        super().__init__()
        layer_count = int(layers)
        if layer_count <= 0:
            raise ValueError(f"layers must be positive, got {layer_count}.")

        hidden_channels = int(out_channels) * int(scale)
        if hidden_channels <= 0:
            raise ValueError(
                f"transparent hidden channels must be positive, got out_channels={out_channels}, scale={scale}."
            )

        self.signal_processing_layers = nn.ModuleList()
        current_channels = int(in_channels)
        for _ in range(layer_count):
            layer = TransparentSignalLayer(
                in_channels=current_channels,
                out_channels=hidden_channels,
                module_names=modules,
                skip_connection=bool(skip_connection),
            )
            self.signal_processing_layers.append(layer)
            current_channels = hidden_channels

        self.channel_for_feature = current_channels
        self.feature_extractor_layers = TransparentFeatureExtractor(features, self.channel_for_feature)
        self.channel_for_classifier = self.channel_for_feature * len(list(features))
        self.resolved_module_names = list(self.signal_processing_layers[0].module_names)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        hidden = to_bcl(x).permute(0, 2, 1)
        for layer in self.signal_processing_layers:
            hidden = layer(hidden)
        return self.feature_extractor_layers(hidden)
