#!/usr/bin/env python
"""TF_MultiProtoDG Quick Experiment Runner

Runs multiple configuration candidates sequentially. For each experiment, writes
a derived YAML config and invokes Pipeline_01_default directly.

Usage:
  conda activate phmbench
  python scripts/tf_multi_proto_dg_quick_exp.py --episodes 3 --epochs 20
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

BASE_CONFIG = "configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml"

# ── Experiment Definitions ──────────────────────────────────────────────
# Each experiment has a description and a flat dict of dot-separated overrides.

EXPERIMENTS = {
    # ── Round 3: Fine-tune temperature and longer training ──
    # R2 winner: anchored_offset + agreement + temp=0.15 → 0.9435±0.0183
    "R3_temp_015_50ep": {
        "description": "R2 winner with 50ep training for convergence",
        "overrides": {
            "model.prototype_residual_score_mode": "anchored_offset",
            "model.prototype_residual_logit_mode": "agreement",
            "model.prototype_temperature": 0.15,
            "model.prototype_assignment_temperature": 0.15,
        },
    },
    "R3_temp_018_20ep": {
        "description": "Wider temperature (0.18) - even smoother routing",
        "overrides": {
            "model.prototype_residual_score_mode": "anchored_offset",
            "model.prototype_residual_logit_mode": "agreement",
            "model.prototype_temperature": 0.18,
            "model.prototype_assignment_temperature": 0.18,
        },
    },
    "R3_temp_015_cw_008": {
        "description": "temp=0.15 + slightly lower cw (0.08 vs 0.10)",
        "overrides": {
            "model.prototype_residual_score_mode": "anchored_offset",
            "model.prototype_residual_logit_mode": "agreement",
            "model.prototype_temperature": 0.15,
            "model.prototype_assignment_temperature": 0.15,
            "model.proto_contrastive_weight": 0.08,
        },
    },
    "R3_temp_015_assign_only": {
        "description": "temp=0.15 on assignment only, keep class temp=0.12",
        "overrides": {
            "model.prototype_residual_score_mode": "anchored_offset",
            "model.prototype_residual_logit_mode": "agreement",
            "model.prototype_assignment_temperature": 0.15,
        },
    },
}


def deep_set(d: dict, key: str, value):
    """Set a nested dict value using dot-separated key."""
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def load_base_config() -> dict:
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)


def make_experiment_config(exp: dict, episodes: int, epochs: int) -> dict:
    cfg = load_base_config()
    for key, value in exp["overrides"].items():
        deep_set(cfg, key, value)
    cfg["environment"]["iterations"] = episodes
    cfg["trainer"]["num_epochs"] = epochs
    cfg["trainer"]["patience"] = epochs
    return cfg


def collect_results(output_dir: str) -> dict:
    """Scan output_dir for test_result CSV files and aggregate."""
    import csv as csv_mod
    results = {"iterations": []}
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname.startswith("test_result_") and fname.endswith(".csv"):
                try:
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        reader = csv_mod.DictReader(f)
                        for row in reader:
                            acc = float(row.get("test_acc_RM_023_HIT23",
                                                 row.get("test_acc", 0)))
                            proto_nce = float(row.get("test_proto_nce", 0))
                            total_loss = float(row.get("test_total_loss", 0))
                            results["iterations"].append({
                                "path": root,
                                "file": fname,
                                "test_acc": acc,
                                "test_proto_nce": proto_nce,
                                "test_total_loss": total_loss,
                            })
                except Exception:
                    pass
            elif fname == "metrics.json":
                try:
                    with open(os.path.join(root, fname)) as f:
                        m = json.load(f)
                    results["iterations"].append({
                        "path": root,
                        "test_acc": m.get("test_acc"),
                        "test_total_loss": m.get("test_total_loss"),
                    })
                except Exception:
                    pass

    accs = [r["test_acc"] for r in results["iterations"]
            if r.get("test_acc") is not None]
    if accs:
        import numpy as np
        results["mean_acc"] = float(np.mean(accs))
        results["std_acc"] = float(np.std(accs))
        results["min_acc"] = float(np.min(accs))
        results["max_acc"] = float(np.max(accs))
    return results


def run_one(name: str, exp: dict, episodes: int, epochs: int,
            base_output: str) -> dict:
    print(f"\n{'='*70}")
    print(f"  [{name}] {exp['description']}")
    print(f"  Episodes={episodes}, Epochs={epochs}")
    print(f"{'='*70}")

    cfg = make_experiment_config(exp, episodes, epochs)
    exp_dir = os.path.join(base_output, name)
    os.makedirs(exp_dir, exist_ok=True)
    cfg["environment"]["output_dir"] = exp_dir

    cfg_path = os.path.join(exp_dir, "_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    from types import SimpleNamespace
    from src.Pipeline_01_default import pipeline

    args = SimpleNamespace(
        config=cfg_path,
        config_path=cfg_path,
        notes=f"Quick exp: {name}",
        override=None,
        local_config=None,
    )

    t0 = time.time()
    try:
        pipeline(args)
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "error": str(e), "duration_s": time.time() - t0}

    duration = time.time() - t0
    res = collect_results(exp_dir)
    res["name"] = name
    res["duration_s"] = duration

    if "mean_acc" in res:
        print(f"\n  >> {name}: acc={res['mean_acc']:.4f} ± {res['std_acc']:.4f} "
              f"[{res['min_acc']:.4f}, {res['max_acc']:.4f}]  ({duration:.0f}s)")
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output-dir", type=str,
                    default="results/demo/tf_multi_proto_dg_exp")
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Subset of experiment names")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    names = args.experiments or list(EXPERIMENTS.keys())
    all_res = []

    for name in names:
        if name not in EXPERIMENTS:
            print(f"WARNING: unknown experiment '{name}'")
            continue
        r = run_one(name, EXPERIMENTS[name], args.episodes, args.epochs,
                     args.output_dir)
        all_res.append(r)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    hdr = f"{'Experiment':<45} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_res:
        n = r["name"]
        if "mean_acc" in r:
            print(f"{n:<45} {r['mean_acc']:>6.4f} {r['std_acc']:>6.4f} "
                  f"{r['min_acc']:>6.4f} {r['max_acc']:>6.4f}")
        else:
            print(f"{n:<45} ERROR: {r.get('error','?')[:30]}")

    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
