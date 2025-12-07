"""Transparent Statistical Prototypical Network (TSPN) with SPD geometry and prototype memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TSPN import TransparentSignalFeatureExtractor as TSFExtractor
from ..modules import (
    PhysicsStatSPDConfig,
    PhysicsStatSPDLayer,
    OrthogonalProjector,
    OrthogonalProjectorConfig,
    ContrastiveHead,
    ClassifierHead,
    PrototypeMemory,
    PrototypeMemoryConfig,
)
import warnings


@dataclass
class LossWeights:
    lambda_proto: float = 1.0
    lambda_var: float = 0.0
    lambda_ortho: float = 1e-3
    lambda_phys: float = 0.0
    temperature: float = 0.2


@dataclass
class HeadConfig:
    contrast_dim: int = 128


@dataclass
class ReducerConfig:
    dim: int


@dataclass
class TSPNContrastiveConfig:
    spd: PhysicsStatSPDConfig
    projector: OrthogonalProjectorConfig
    head: HeadConfig
    loss: LossWeights
    prototype: PrototypeMemoryConfig
    reducer: ReducerConfig


def _resolve_num_classes(args: Any) -> int:
    num_classes = getattr(args, "num_classes", None)
    if isinstance(num_classes, dict):
        if not num_classes:
            raise ValueError("num_classes mapping is empty")
        return int(max(num_classes.values()))
    if isinstance(num_classes, (list, tuple)):
        if not num_classes:
            raise ValueError("num_classes sequence is empty")
        return int(num_classes[0])
    if num_classes is None:
        raise ValueError("Model configuration missing num_classes")
    return int(num_classes)


def _to_cfg(args: Any, feat_dim: int, num_classes: int) -> TSPNContrastiveConfig:
    raw = getattr(args, "tspn", None) or getattr(args, "contrastive", None)
    spd_raw = getattr(raw, "spd", None)
    projector_raw = getattr(raw, "projector", None)
    head_raw = getattr(raw, "head", None)
    loss_raw = getattr(raw, "loss", None)
    proto_raw = getattr(raw, "prototype_memory", None)
    reducer_raw = getattr(raw, "reducer", None)

    reducer_dim = int(getattr(reducer_raw, "dim", max(32, feat_dim // 4)))

    spd_cfg = PhysicsStatSPDConfig(
        feat_dim=reducer_dim,
        eps=float(getattr(spd_raw, "eps", 1e-3)),
        momentum=float(getattr(spd_raw, "momentum", 0.1)),
        block_size=int(getattr(spd_raw, "block_size", 0)),
    )
    projector_cfg = OrthogonalProjectorConfig(
        in_dim=spd_cfg.feat_dim + spd_cfg.feat_dim * (spd_cfg.feat_dim + 1) // 2,
        proj_dim=int(getattr(projector_raw, "proj_dim", min(64, spd_cfg.feat_dim * 2))),
        reorth_every=int(getattr(projector_raw, "reorth_every", 10)),
    )
    head_cfg = HeadConfig(contrast_dim=int(getattr(head_raw, "contrast_dim", 128)))
    loss_cfg = LossWeights(
        lambda_proto=float(getattr(loss_raw, "lambda_proto", 1.0)),
        lambda_var=float(getattr(loss_raw, "lambda_var", 0.0)),
        lambda_ortho=float(getattr(loss_raw, "lambda_ortho", 1e-3)),
        lambda_phys=float(getattr(loss_raw, "lambda_phys", 0.0)),
        temperature=float(getattr(loss_raw, "temperature", getattr(raw, "temperature", 0.2))),
    )
    proto_cfg = PrototypeMemoryConfig(
        num_classes=num_classes,
        feat_dim=head_cfg.contrast_dim,
        num_domains=getattr(proto_raw, "num_domains", None),
        momentum=float(getattr(proto_raw, "momentum", 0.99)),
    )
    reducer_cfg = ReducerConfig(dim=reducer_dim)
    return TSPNContrastiveConfig(
        spd=spd_cfg,
        projector=projector_cfg,
        head=head_cfg,
        loss=loss_cfg,
        prototype=proto_cfg,
        reducer=reducer_cfg,
    )


class Model(nn.Module):
    """Transparent Statistical Prototypical Network."""

    def __init__(self, args: Any, metadata: Optional[Any] = None) -> None:
        super().__init__()
        if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            args.device = "cpu"
        self.args = args
        self.extractor = TSFExtractor(args, metadata)
        self.raw_feature_dim = self.extractor.channel_for_classifier
        self.num_classes = _resolve_num_classes(args)

        cfg = _to_cfg(args, self.raw_feature_dim, self.num_classes)
        self.cfg = cfg

        self.reducer = nn.Linear(self.extractor.channel_for_classifier, cfg.reducer.dim)
        self.spd_layer = PhysicsStatSPDLayer(cfg.spd)
        self.projector = OrthogonalProjector(cfg.projector)
        self.contrastive_head = ContrastiveHead(cfg.projector.in_dim, cfg.head.contrast_dim)
        self.classifier = ClassifierHead(cfg.projector.in_dim, self.num_classes)
        self.prototype_memory = PrototypeMemory(cfg.prototype)

    @property
    def feature_dim(self) -> int:
        return self.cfg.reducer.dim

    @property
    def embedding_dim(self) -> int:
        return self.cfg.projector.in_dim

    def forward(self, x: torch.Tensor, *_, return_dict: bool = False, **__) -> torch.Tensor | Dict[str, torch.Tensor]:
        h_raw = self.extractor(x)
        h = self.reducer(h_raw)
        v = self.spd_layer(h)
        z_phys, z_spur = self.projector(v)
        logits = self.classifier(z_phys)
        u = self.contrastive_head(z_phys)
        payload = {
            "h_raw": h_raw,
            "h": h,
            "spd_vec": v[:, self.feature_dim :],
            "v": v,
            "z_phys": z_phys,
            "z_spur": z_spur,
            "u": u,
            "logits": logits,
        }
        if return_dict:
            return payload
        return logits

    def reorthogonalize(self) -> None:
        self.projector.reorthogonalize()

    def ortho_loss(self) -> torch.Tensor:
        return self.projector.ortho_loss()

    def contrastive_temperature(self) -> float:
        return self.cfg.loss.temperature
