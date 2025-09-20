"""Explainability helpers for TSPN few-shot domain generalisation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch


def _sanitize(value: Any) -> str:
    text = str(value)
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in text)


def _stack_embeddings(entries: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [t.unsqueeze(0) if t.ndim == 1 else t for t in entries]
    return torch.cat(tensors, dim=0)


def export_tspn_explainability(
    *,
    stage: str,
    cached_batches: List[Dict[str, Any]],
    metadata,
    config,
    output_dir: str,
) -> None:
    """Persist per-domain attribution summaries for TSPN few-shot runs.

    Parameters
    ----------
    stage:
        Evaluation stage (e.g. ``"val"`` or ``"test"``).
    cached_batches:
        List of dictionaries produced by the Lightning task containing
        ``embeddings`` tensors, ``file_ids`` lists, and ``labels`` tensors.
    metadata:
        Metadata accessor used to resolve domain/system names.
    config:
        Explainability configuration namespace.
    output_dir:
        Root experiment directory supplied via the config.
    """
    if not cached_batches:
        return

    summary_cfg = getattr(config, "summary", None)
    top_k = int(getattr(summary_cfg, "top_k_features", 10) or 10)
    save_embeddings = bool(getattr(summary_cfg, "save_embeddings", False))

    output_subdir = getattr(config, "output_subdir", "explainability")
    stage_dir = Path(output_dir).expanduser() / output_subdir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate embeddings per (system, domain).
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for batch in cached_batches:
        embeddings = batch["embeddings"]
        file_ids = batch["file_ids"]
        labels = batch["labels"]

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings
        else:
            embeddings = torch.stack(embeddings)

        for idx, fid in enumerate(file_ids):
            meta = metadata[fid]
            system = _sanitize(meta.get("Dataset_id", "unknown"))
            domain = _sanitize(meta.get("Domain_id", "unknown"))
            key = (system, domain)
            grouped.setdefault(key, {"embeddings": [], "labels": []})
            grouped[key]["embeddings"].append(embeddings[idx].cpu())
            grouped[key]["labels"].append(int(labels[idx]))

    for (system, domain), payload in grouped.items():
        emb_tensor = _stack_embeddings(payload["embeddings"]).cpu()
        label_tensor = torch.as_tensor(payload["labels"])
        feature_count = emb_tensor.size(1)
        mean_abs = emb_tensor.abs().mean(dim=0)
        mean = emb_tensor.mean(dim=0)
        std = emb_tensor.std(dim=0, unbiased=False)

        k = min(top_k, feature_count)
        top_values, top_indices = torch.topk(mean_abs, k)

        rows = []
        for rank, (score, index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature_index": index,
                    "mean_abs": score,
                    "mean": mean[index].item(),
                    "std": std[index].item(),
                    "support": emb_tensor.size(0),
                }
            )

        df = pd.DataFrame(rows)
        filename = stage_dir / f"system_{system}_domain_{domain}_top_features.csv"
        df.to_csv(filename, index=False)

        # Store label histogram for interpretability context.
        label_counts = pd.Series(label_tensor.numpy()).value_counts().sort_index()
        label_counts.to_csv(
            stage_dir / f"system_{system}_domain_{domain}_label_histogram.csv",
            header=["count"],
        )

        if save_embeddings:
            emb_path = stage_dir / f"system_{system}_domain_{domain}_embeddings.pt"
            torch.save({"embeddings": emb_tensor, "labels": label_tensor}, emb_path)
