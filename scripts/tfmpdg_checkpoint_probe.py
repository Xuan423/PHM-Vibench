#!/usr/bin/env python3
"""Probe TF_MultiProtoDG checkpoints with per-class decision-path diagnostics.

The script rebuilds the dataset split from checkpoint hyper-parameters, loads the
network weights, and exports compact per-class/global statistics.  It is meant
for low-I/O root-cause analysis; it does not train or write large tensors.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs.config_utils import load_config, transfer_namespace
from src.data_factory import build_data
from src.model_factory import build_model


def _set_nested(ns: Any, dotted_key: str, value: Any) -> None:
    target = ns
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part) or getattr(target, part) is None:
            setattr(target, part, SimpleNamespace())
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _apply_checkpoint_hparams(config: Any, hparams: dict[str, Any]) -> None:
    for key in (
        "seed",
        "current_seed",
        "source_domain_id",
        "target_domain_id",
        "target_system_id",
        "label_smoothing",
    ):
        if key in hparams:
            section = "environment" if key in {"seed", "current_seed"} else "task"
            if key == "current_seed":
                _set_nested(config, "environment.seed", int(hparams[key]))
            else:
                _set_nested(config, f"{section}.{key}", hparams[key])

    model_keys = {
        "input_length",
        "time_operators",
        "freq_operators",
        "time_indicators",
        "freq_indicators",
        "time_patch_count",
        "time_patch_mode",
        "time_patch_width",
        "freq_band_count",
        "freq_band_mode",
        "freq_band_width",
        "region_sampling_mode",
        "region_sampling_seed_offset",
        "transparent_backbone_enabled",
        "transparent_backbone_weight",
        "transparent_time_dual_basis_enabled",
        "transparent_backbone_layers",
        "transparent_backbone_modules",
        "transparent_backbone_features",
        "transparent_backbone_freq_modules",
        "transparent_backbone_freq_features",
        "structured_global_enabled",
        "role_dim",
        "concept_dim",
        "num_prototypes_per_class",
        "prototype_temperature",
        "prototype_assignment_temperature",
        "prototype_class_pool_mode",
        "prototype_class_anchor_mode",
        "prototype_class_anchor_memory_weight",
        "prototype_class_anchor_memory_momentum",
        "prototype_anchor_input",
        "prototype_concept_input",
        "prototype_assignment_input",
        "local_prototype_concept_source",
        "local_anomaly_residual_mode",
        "local_anomaly_gate_mode",
        "semantic_residual_init",
        "prototype_residual_score_mode",
        "prototype_residual_logit_weight",
        "prototype_residual_warmup_epochs",
        "prototype_residual_logit_mode",
        "proto_contrastive_temperature",
        "proto_contrastive_positive_mode",
        "proto_contrastive_weight",
        "proto_contrastive_use_scaled_scores",
        "prototype_logit_scale_init",
        "prototype_init_mode",
        "prototype_init_scale",
        "cooperative_prototypes_enabled",
        "freq_semantic_mode",
        "tf_interaction_mode",
        "concept_layout",
        "prototype_anchor_score_mode",
        "prototype_identity_mode",
        "global_semantic_refiner_mode",
        "concept_segment_norm_enabled",
        "role_nonneg",
    }
    for key in model_keys:
        if key in hparams:
            _set_nested(config, f"model.{key}", hparams[key])
    prototype_hparams = hparams.get("prototype")
    if hasattr(prototype_hparams, "items"):
        prototype_items = prototype_hparams.items()
    elif prototype_hparams is not None and hasattr(prototype_hparams, "__dict__"):
        prototype_items = vars(prototype_hparams).items()
    else:
        prototype_items = ()
    for key, value in prototype_items:
        _set_nested(config, f"model.prototype.{key}", value)

    classifier_hparams = hparams.get("classifier")
    if hasattr(classifier_hparams, "items"):
        classifier_items = classifier_hparams.items()
    elif classifier_hparams is not None and hasattr(classifier_hparams, "__dict__"):
        classifier_items = vars(classifier_hparams).items()
    else:
        classifier_items = ()
    for key, value in classifier_items:
        _set_nested(config, f"model.classifier.{key}", value)

    _set_nested(config, "data.num_workers", 0)
    _set_nested(config, "data.pin_memory", False)
    _set_nested(config, "trainer.num_workers", 0)
    _set_nested(config, "trainer.pin_memory", False)


def _strip_network_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[len("network.") :] if key.startswith("network.") else key): value
        for key, value in state_dict.items()
        if key.startswith("network.")
    }


def _focus(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2 or weights.shape[1] <= 1:
        return torch.ones(weights.shape[0], 1, device=weights.device, dtype=weights.dtype)
    focus = weights.pow(2).sum(dim=1, keepdim=True)
    min_focus = 1.0 / float(weights.shape[1])
    return ((focus - min_focus) / max(1.0 - min_focus, 1e-6)).clamp(0.0, 1.0)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _segment_true_margins(
    anchor_h: torch.Tensor | None,
    class_anchor: torch.Tensor | None,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not torch.is_tensor(anchor_h)
        or not torch.is_tensor(class_anchor)
        or anchor_h.ndim != 2
        or class_anchor.ndim != 2
        or anchor_h.shape[-1] != class_anchor.shape[-1]
        or int(anchor_h.shape[-1]) % 4 != 0
    ):
        nan = torch.full((labels.shape[0], 4), float("nan"), device=labels.device)
        return nan, nan
    dim = int(anchor_h.shape[-1]) // 4
    h = F.normalize(anchor_h, dim=-1).view(anchor_h.shape[0], 4, dim)
    anchors = F.normalize(class_anchor, dim=-1).view(class_anchor.shape[0], 4, dim)
    scores = torch.einsum("bsd,csd->bcs", h, anchors)
    margins = []
    true_scores = []
    for seg_id in range(4):
        seg_scores = scores[:, :, seg_id]
        margins.append(_true_margin(seg_scores, labels))
        true_scores.append(seg_scores.gather(1, labels.view(-1, 1)).squeeze(1))
    return torch.stack(margins, dim=1), torch.stack(true_scores, dim=1)


@torch.no_grad()
def probe_checkpoint(
    config_path: str,
    checkpoint: str,
    device: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    config = load_config(config_path)
    _apply_checkpoint_hparams(config, hparams)
    seed = int(hparams.get("current_seed", hparams.get("seed", getattr(config.environment, "seed", 42))))
    seed_everything(seed)

    args_data = transfer_namespace(config.data)
    args_task = transfer_namespace(config.task)
    args_model = transfer_namespace(config.model)
    args_model.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"

    data_factory = build_data(args_data, args_task)
    model = build_model(args_model, metadata=data_factory.get_metadata())
    missing, unexpected = model.load_state_dict(_strip_network_prefix(ckpt["state_dict"]), strict=False)
    if unexpected:
        print(f"[WARN] unexpected network keys: {len(unexpected)}")
    if missing:
        print(f"[WARN] missing network keys: {len(missing)}")

    run_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(run_device)
    model.eval()

    rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(data_factory.get_dataloader(split)):
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(run_device)
        batch["_file_ids_raw"] = batch.get("file_id")
        out = model.forward_with_batch(batch, epoch=int(ckpt.get("epoch", 0)))
        y = batch["y"].long()
        logits = out["logits"]
        anchor = out.get("anchor_scores")
        residual = out.get("pooled_residual_scores")
        if not torch.is_tensor(anchor):
            anchor = logits
        if not torch.is_tensor(residual):
            residual = logits
        proto_probs = out.get("target_proto_probs")
        if not torch.is_tensor(proto_probs):
            proto_probs = torch.empty(y.shape[0], 0, device=run_device)
        w_t = out.get("w_t_local_anomaly")
        w_f = out.get("w_f_local_anomaly")
        time_focus = _focus(w_t).squeeze(1) if torch.is_tensor(w_t) else torch.full_like(y.float(), float("nan"))
        freq_focus = _focus(w_f).squeeze(1) if torch.is_tensor(w_f) else torch.full_like(y.float(), float("nan"))
        h_anchor = out.get("h_anchor_global")
        h_anchor_raw = out.get("h_anchor_raw_global")
        h_local_proto = out.get("h_proto_local")
        h_fused = out.get("h")
        class_anchor = out.get("class_anchor")
        if torch.is_tensor(h_anchor) and torch.is_tensor(h_fused):
            anchor_fused_cos = F.cosine_similarity(h_anchor, h_fused, dim=-1)
            fused_delta = (h_anchor - h_fused).norm(dim=-1)
        else:
            anchor_fused_cos = torch.full_like(y.float(), float("nan"))
            fused_delta = torch.full_like(y.float(), float("nan"))
        seg_margins, seg_true_scores = _segment_true_margins(h_anchor, class_anchor, y)
        if torch.is_tensor(h_anchor_raw) and torch.is_tensor(class_anchor):
            raw_anchor_scores = torch.einsum(
                "bd,cd->bc",
                F.normalize(h_anchor_raw, dim=-1),
                F.normalize(class_anchor, dim=-1),
            )
            raw_anchor_margin = _true_margin(raw_anchor_scores, y)
            raw_anchor_pred = raw_anchor_scores.argmax(dim=1)
        else:
            raw_anchor_margin = torch.full_like(y.float(), float("nan"))
            raw_anchor_pred = torch.full_like(y, -1)
        if torch.is_tensor(h_local_proto) and torch.is_tensor(class_anchor):
            local_anchor_scores = torch.einsum(
                "bd,cd->bc",
                F.normalize(h_local_proto, dim=-1),
                F.normalize(class_anchor, dim=-1),
            )
            local_anchor_margin = _true_margin(local_anchor_scores, y)
            local_anchor_pred = local_anchor_scores.argmax(dim=1)
        else:
            local_anchor_margin = torch.full_like(y.float(), float("nan"))
            local_anchor_pred = torch.full_like(y, -1)
        anchor_margin = _true_margin(anchor, y)
        final_margin = _true_margin(logits, y)
        residual_margin = _true_margin(residual, y)
        final_pred = logits.argmax(dim=1)
        anchor_pred = anchor.argmax(dim=1)
        residual_pred = residual.argmax(dim=1)
        proto_top_prob = proto_probs.max(dim=-1).values if proto_probs.numel() else torch.full_like(y.float(), float("nan"))
        for idx in range(y.shape[0]):
            rows.append(
                {
                    "batch": batch_idx,
                    "label": int(y[idx].item()),
                    "final_correct": float(final_pred[idx].eq(y[idx]).item()),
                    "anchor_correct": float(anchor_pred[idx].eq(y[idx]).item()),
                    "raw_anchor_correct": float(raw_anchor_pred[idx].eq(y[idx]).item()),
                    "local_anchor_correct": float(local_anchor_pred[idx].eq(y[idx]).item()),
                    "residual_correct": float(residual_pred[idx].eq(y[idx]).item()),
                    "final_pred": int(final_pred[idx].item()),
                    "anchor_pred": int(anchor_pred[idx].item()),
                    "raw_anchor_pred": int(raw_anchor_pred[idx].item()),
                    "local_anchor_pred": int(local_anchor_pred[idx].item()),
                    "residual_pred": int(residual_pred[idx].item()),
                    "final_margin": float(final_margin[idx].item()),
                    "anchor_margin": float(anchor_margin[idx].item()),
                    "raw_anchor_margin": float(raw_anchor_margin[idx].item()),
                    "local_anchor_margin": float(local_anchor_margin[idx].item()),
                    "residual_margin": float(residual_margin[idx].item()),
                    "proto_top_prob": float(proto_top_prob[idx].item()),
                    "time_focus": float(time_focus[idx].item()),
                    "freq_focus": float(freq_focus[idx].item()),
                    "anchor_fused_cos": float(anchor_fused_cos[idx].item()),
                    "anchor_fused_delta": float(fused_delta[idx].item()),
                    "seg_time_margin": float(seg_margins[idx, 0].item()),
                    "seg_freq_margin": float(seg_margins[idx, 1].item()),
                    "seg_tf_margin": float(seg_margins[idx, 2].item()),
                    "seg_gap_margin": float(seg_margins[idx, 3].item()),
                    "seg_time_true": float(seg_true_scores[idx, 0].item()),
                    "seg_freq_true": float(seg_true_scores[idx, 1].item()),
                    "seg_tf_true": float(seg_true_scores[idx, 2].item()),
                    "seg_gap_true": float(seg_true_scores[idx, 3].item()),
                }
            )
    try:
        data_factory.data.close()
    except Exception:
        pass

    sample_df = pd.DataFrame(rows)
    summary_rows = []
    for label, group in sample_df.groupby("label", sort=True):
        final_hist = group["final_pred"].value_counts().sort_index()
        anchor_hist = group["anchor_pred"].value_counts().sort_index()
        raw_anchor_hist = group["raw_anchor_pred"].value_counts().sort_index()
        local_anchor_hist = group["local_anchor_pred"].value_counts().sort_index()
        residual_hist = group["residual_pred"].value_counts().sort_index()
        summary_rows.append(
            {
                "label": int(label),
                "count": int(len(group)),
                "final_acc": group["final_correct"].mean(),
                "anchor_acc": group["anchor_correct"].mean(),
                "raw_anchor_acc": group["raw_anchor_correct"].mean(),
                "local_anchor_acc": group["local_anchor_correct"].mean(),
                "residual_acc": group["residual_correct"].mean(),
                "final_margin": group["final_margin"].mean(),
                "anchor_margin": group["anchor_margin"].mean(),
                "raw_anchor_margin": group["raw_anchor_margin"].mean(),
                "local_anchor_margin": group["local_anchor_margin"].mean(),
                "residual_margin": group["residual_margin"].mean(),
                "proto_top_prob": group["proto_top_prob"].mean(),
                "time_focus": group["time_focus"].mean(),
                "freq_focus": group["freq_focus"].mean(),
                "anchor_fused_cos": group["anchor_fused_cos"].mean(),
                "anchor_fused_delta": group["anchor_fused_delta"].mean(),
                "seg_time_margin": group["seg_time_margin"].mean(),
                "seg_freq_margin": group["seg_freq_margin"].mean(),
                "seg_tf_margin": group["seg_tf_margin"].mean(),
                "seg_gap_margin": group["seg_gap_margin"].mean(),
                "seg_time_true": group["seg_time_true"].mean(),
                "seg_freq_true": group["seg_freq_true"].mean(),
                "seg_tf_true": group["seg_tf_true"].mean(),
                "seg_gap_true": group["seg_gap_true"].mean(),
                "final_pred_hist": ";".join(f"{int(k)}:{int(v)}" for k, v in final_hist.items()),
                "anchor_pred_hist": ";".join(f"{int(k)}:{int(v)}" for k, v in anchor_hist.items()),
                "raw_anchor_pred_hist": ";".join(
                    f"{int(k)}:{int(v)}" for k, v in raw_anchor_hist.items()
                ),
                "local_anchor_pred_hist": ";".join(
                    f"{int(k)}:{int(v)}" for k, v in local_anchor_hist.items()
                ),
                "residual_pred_hist": ";".join(
                    f"{int(k)}:{int(v)}" for k, v in residual_hist.items()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    meta = {
        "checkpoint": str(checkpoint_path),
        "run": checkpoint_path.parents[1].name,
        "iter": checkpoint_path.parent.name,
        "seed": seed,
        "source": hparams.get("source_domain_id"),
        "target": hparams.get("target_domain_id"),
        "split": split,
        "epoch": int(ckpt.get("epoch", -1)),
        "sample_acc": float(sample_df["final_correct"].mean()),
        "anchor_acc": float(sample_df["anchor_correct"].mean()),
        "residual_acc": float(sample_df["residual_correct"].mean()),
    }
    return summary, meta


def _true_margin(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(scores.device).long()
    true_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)
    mask = F.one_hot(labels, num_classes=scores.shape[1]).bool()
    neg = scores.masked_fill(mask, torch.finfo(scores.dtype).min).max(dim=1).values
    return true_scores - neg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--run", action="append", default=[], help="Run directory containing iter_*/.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    checkpoints = list(args.checkpoint)
    for run in args.run:
        checkpoints.extend(sorted(glob.glob(os.path.join(run, "iter_*", "*.ckpt"))))
    if not checkpoints:
        raise SystemExit("Provide --checkpoint or --run.")

    all_rows = []
    for checkpoint in checkpoints:
        summary, meta = probe_checkpoint(args.config, checkpoint, args.device, args.split)
        print(
            f"\n# {meta['run']} {meta['iter']} seed={meta['seed']} "
            f"{meta['source']}->{meta['target']} split={meta['split']} epoch={meta['epoch']} "
            f"acc={meta['sample_acc']:.4f} anchor={meta['anchor_acc']:.4f}"
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        run_rows = summary.copy()
        for key, value in meta.items():
            out_key = key if key not in run_rows.columns else f"meta_{key}"
            run_rows[out_key] = str(value)
        all_rows.append(run_rows)

    if args.out and all_rows:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
