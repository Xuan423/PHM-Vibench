"""Utilities for capturing projector diagnostics across training stages."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ProjectorSnapshot:
    stage: str
    event: str
    timestamp: float
    state_path: str
    matrix_path: str
    fro_norm: float
    spectral_norm: float
    feature_dim: int
    axis_dim: int
    delta_fro: Optional[float] = None
    delta_cosine: Optional[float] = None


def snapshot_projector(
    network: Any,
    directory: Path,
    *,
    stage: str,
    event: str,
    compare_manifest: Optional[str] = None,
) -> Optional[str]:
    projector = getattr(network, "physics_projector", None)
    if projector is None:
        return None

    explain_fn = getattr(network, "explain_projection_matrix", None)
    if not callable(explain_fn):
        return None
    projection = explain_fn()
    if projection is None:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    projection_cpu = projection.detach().cpu()
    state_path = directory / f"{event}_projector_state.pt"
    torch.save(projector.state_dict(), state_path)

    matrix_path = directory / f"{event}_projection.pt"
    torch.save({"projection": projection_cpu}, matrix_path)

    delta_stats = _compute_delta(projection_cpu, compare_manifest)

    snapshot = ProjectorSnapshot(
        stage=stage,
        event=event,
        timestamp=time.time(),
        state_path=str(state_path),
        matrix_path=str(matrix_path),
        fro_norm=float(torch.linalg.norm(projection_cpu, ord="fro").item()),
        spectral_norm=float(torch.linalg.norm(projection_cpu, ord=2).item()),
        feature_dim=int(projection_cpu.shape[0]),
        axis_dim=int(projection_cpu.shape[1]),
        delta_fro=delta_stats.get("delta_fro"),
        delta_cosine=delta_stats.get("delta_cosine"),
    )

    manifest_path = directory / f"{event}_snapshot.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(snapshot), handle, indent=2, ensure_ascii=False)
    return str(manifest_path)


def _compute_delta(current: torch.Tensor, compare_manifest: Optional[str]) -> Dict[str, float]:
    if compare_manifest is None:
        return {}
    compare_path = Path(compare_manifest)
    if not compare_path.exists():
        return {}
    with compare_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matrix_path = payload.get("matrix_path")
    if not matrix_path:
        return {}
    stored = torch.load(matrix_path)
    previous = stored.get("projection")
    if previous is None:
        return {}
    previous = previous.to(current)

    diff = current - previous
    delta_fro = torch.linalg.norm(diff, ord="fro").item()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            current.view(-1).unsqueeze(0), previous.view(-1).unsqueeze(0), dim=-1
        )[0].item()
    )
    return {"delta_fro": float(delta_fro), "delta_cosine": cosine}


def load_snapshot_manifest(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
