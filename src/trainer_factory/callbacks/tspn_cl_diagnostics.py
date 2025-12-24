"""Diagnostics exports for TSPN_CL.

Exports paper-friendly CSV/MD artifacts for Top-K selection and metric sparsity.
The callback is designed to be safe to attach globally and remain near-zero
overhead when diagnostics are disabled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class TSPNCLDiagnosticsCallback(pl.Callback):
    """Export Top-K and sparsity diagnostics into the run directory.

    Enablement
    ----------
    The callback writes outputs only when ``pl_module.network.export_diagnostics`` is truthy.
    """

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.diagnostics_dir = self.output_dir / "diagnostics"
        self.topk_csv = self.diagnostics_dir / "topk_report.csv"
        self.sparsity_csv = self.diagnostics_dir / "sparsity_report.csv"
        self.sparsity_md = self.diagnostics_dir / "sparsity_report.md"
        self.prototypes_json = self.diagnostics_dir / "prototypes_report.json"
        self._prev_topk: Optional[set[int]] = None

    def _enabled(self, pl_module: pl.LightningModule) -> bool:
        network = getattr(pl_module, "network", None)
        return bool(getattr(network, "export_diagnostics", False))

    def _append_csv_row(self, path: Path, header: str, row: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(header + "\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")

    def _write_sparsity_md_once(self) -> None:
        if self.sparsity_md.exists():
            return
        self.sparsity_md.parent.mkdir(parents=True, exist_ok=True)
        self.sparsity_md.write_text(
            "# Sparsity Diagnostics (TSPN_CL)\n\n"
            "- `sparsity_coeff_row`: row-wise L1 penalty on `V` (encourages sparse rows / sparse concepts)\n"
            "- `sparsity_coeff_col`: column-wise L2 penalty on `V` (shrinks per-feature contribution)\n",
            encoding="utf-8",
        )

    def _append_json_record(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: list[Dict[str, Any]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    data = loaded
            except Exception:
                data = []
        data.append(record)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._enabled(pl_module):
            return

        network = pl_module.network
        epoch = int(getattr(trainer, "current_epoch", 0))

        # ---- Top-K report ----
        selector = getattr(network, "topk_selector", None)
        feature_dim = getattr(network, "feature_dim", None)
        if selector is not None and feature_dim is not None:
            idx = selector.get_indices(num_features=int(feature_dim), device=torch.device("cpu"))
            idx_list = [int(i) for i in idx.detach().cpu().tolist()]
            idx_set = set(idx_list)

            overlap = None
            changed = None
            if self._prev_topk is not None and len(idx_set) > 0:
                inter = len(idx_set.intersection(self._prev_topk))
                overlap = float(inter) / float(len(idx_set))
                changed = int(len(idx_set) - inter)
            self._prev_topk = idx_set

            running_score = getattr(selector, "running_score", None)
            score_stats: Dict[str, Any] = {}
            if running_score is not None and running_score.numel() == int(feature_dim):
                scores = running_score.detach().cpu()
                sel_scores = scores[idx].float()
                score_stats = {
                    "score_min": float(sel_scores.min().item()),
                    "score_mean": float(sel_scores.mean().item()),
                    "score_max": float(sel_scores.max().item()),
                }

            header = (
                "epoch,top_k,frozen,overlap_rate,changed_count,indices,"
                "score_min,score_mean,score_max"
            )
            # Avoid commas in CSV fields by encoding indices as ';'-separated integers.
            indices_str = ";".join(str(i) for i in idx_list)
            row = ",".join(
                [
                    str(epoch),
                    str(len(idx_list)),
                    str(int(bool(getattr(selector, "frozen", False)))),
                    "" if overlap is None else f"{overlap:.6f}",
                    "" if changed is None else str(changed),
                    indices_str,
                    str(score_stats.get("score_min", "")),
                    str(score_stats.get("score_mean", "")),
                    str(score_stats.get("score_max", "")),
                ]
            )
            self._append_csv_row(self.topk_csv, header=header, row=row)

        # ---- Sparsity report ----
        metric = getattr(network, "metric", None)
        if metric is not None and hasattr(metric, "weight"):
            V = metric.weight.detach()
            row_l1 = torch.norm(V, p=1, dim=1).mean().item()
            col_l2 = torch.norm(V, p=2, dim=0).mean().item()
            row_c = float(getattr(network, "sparsity_coeff_row", 0.0))
            col_c = float(getattr(network, "sparsity_coeff_col", 0.0))

            header = "epoch,row_l1_mean,col_l2_mean,sparsity_coeff_row,sparsity_coeff_col,row_pen,col_pen"
            row = ",".join(
                [
                    str(epoch),
                    f"{row_l1:.8f}",
                    f"{col_l2:.8f}",
                    str(row_c),
                    str(col_c),
                    f"{row_c * row_l1:.8f}",
                    f"{col_c * col_l2:.8f}",
                ]
            )
            self._append_csv_row(self.sparsity_csv, header=header, row=row)
            self._write_sparsity_md_once()

        # ---- PrototypeBank report (multi-prototype ProtoNCE) ----
        proto_map = getattr(network, "_proto_head_key_to_bufkey", None)
        if not isinstance(proto_map, dict) or not proto_map:
            return

        heads_report: Dict[str, Any] = {}
        for head_key, bufkey in proto_map.items():
            P = getattr(network, f"_proto_P_{bufkey}", None)
            counts = getattr(network, f"_proto_epoch_counts_{bufkey}", None)
            if P is None or counts is None:
                continue

            P_cpu = P.detach().float().cpu()  # [C, M, d]
            counts_cpu = counts.detach().cpu().long()  # [C, M]
            C, M, _ = P_cpu.shape

            # Occupancy health
            sum_counts = counts_cpu.sum(dim=1)  # [C]
            max_counts = counts_cpu.max(dim=1).values  # [C]
            sum_f = sum_counts.float()
            max_ratio = torch.where(sum_counts > 0, max_counts.float() / sum_f, torch.zeros_like(sum_f))

            denom = sum_f.clamp_min(1.0).unsqueeze(1)
            p = counts_cpu.float() / denom
            entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
            n_eff = torch.exp(entropy)
            n_eff = torch.where(sum_counts > 0, n_eff, torch.zeros_like(n_eff))

            # Pairwise cosine similarities per class
            if M > 1:
                P_norm = F.normalize(P_cpu, p=2, dim=-1)
                pairwise = torch.matmul(P_norm, P_norm.transpose(-1, -2)).clamp(-1.0, 1.0)  # [C,M,M]
                off = pairwise.clone()
                diag = torch.arange(M)
                off[:, diag, diag] = float("-inf")
                max_offdiag = off.view(C, -1).max(dim=1).values
                max_offdiag = torch.where(torch.isfinite(max_offdiag), max_offdiag, torch.zeros_like(max_offdiag))
            else:
                pairwise = torch.ones(C, 1, 1)
                max_offdiag = torch.zeros(C)

            heads_report[str(head_key)] = {
                "num_classes": int(C),
                "M": int(M),
                "max_assignment_ratio": [float(x) for x in max_ratio.tolist()],
                "n_eff": [float(x) for x in n_eff.tolist()],
                "max_offdiag_cos": [float(x) for x in max_offdiag.tolist()],
                "pairwise_cos": pairwise.tolist(),
            }

        if heads_report:
            self._append_json_record(self.prototypes_json, {"epoch": epoch, "heads": heads_report})
