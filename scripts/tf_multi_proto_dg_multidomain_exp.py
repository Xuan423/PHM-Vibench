#!/usr/bin/env python
"""TF_MultiProtoDG Multi-Domain Optimization Experiment Runner

Tests configurations across target domains 0/1/2 with specific accuracy targets:
  Domain 0: >= 95%  |  Domain 1: >= 99%  |  Domain 2: >= 97%

Usage:
  conda activate phmbench
  python scripts/tf_multi_proto_dg_multidomain_exp.py --round baseline
  python scripts/tf_multi_proto_dg_multidomain_exp.py --round R4 --epochs 100 --episodes 5
"""
from __future__ import annotations

import csv as csv_mod
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

BASE_CONFIG = "configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml"

TARGET_DOMAINS = [0, 1, 2]
ACC_TARGETS = {0: 0.95, 1: 0.99, 2: 0.97}

# v3.2 base overrides (anchored_offset + agreement + temp=0.15)
V32_BASE = {
    "model.prototype_residual_score_mode": "anchored_offset",
    "model.prototype_residual_logit_mode": "agreement",
    "model.prototype_temperature": 0.15,
    "model.prototype_assignment_temperature": 0.15,
}

ROUNDS = {
    "baseline": {
        "desc": "v3.2 baseline across all domains",
        "experiments": {
            "v32_baseline": {
                "overrides": dict(V32_BASE),
            },
        },
    },
    "R4": {
        "desc": "Round 4: lr, cw, epochs tuning",
        "experiments": {
            "v32_lr_0005": {
                "overrides": {**V32_BASE, "task.lr": 0.0005},
            },
            "v32_lr_002": {
                "overrides": {**V32_BASE, "task.lr": 0.002},
            },
            "v32_cw_012": {
                "overrides": {**V32_BASE, "model.proto_contrastive_weight": 0.12},
            },
            "v32_cw_006": {
                "overrides": {**V32_BASE, "model.proto_contrastive_weight": 0.06},
            },
            "v32_ls_01": {
                "overrides": {**V32_BASE, "task.label_smoothing": 0.10},
            },
            "v32_ls_005": {
                "overrides": {**V32_BASE, "task.label_smoothing": 0.05},
            },
        },
    },
    "R5": {
        "desc": "Round 5: structural refinements",
        "experiments": {
            "v32_k2": {
                "overrides": {**V32_BASE, "model.num_prototypes_per_class": 2},
            },
            "v32_k4": {
                "overrides": {**V32_BASE, "model.num_prototypes_per_class": 4},
            },
            "v32_scale_8": {
                "overrides": {**V32_BASE, "model.prototype_logit_scale_init": 8.0},
            },
            "v32_scale_4": {
                "overrides": {**V32_BASE, "model.prototype_logit_scale_init": 4.0},
            },
            "v32_balance_01": {
                "overrides": {**V32_BASE, "model.prototype.balance_weight": 0.1},
            },
            "v32_balance_0": {
                "overrides": {**V32_BASE, "model.prototype.balance_weight": 0.0},
            },
        },
    },
    # ── R6: Core architecture diagnosis ──────────────────────────────────
    # Current config has cross_self_mix=1.0 (bypasses cross-evidence attn),
    # structured_global=false, and cw=0.1 (aggressive contrastive).  R6 probes
    # the effect of restoring each component and tuning the contrastive weight.
    "R6": {
        "desc": "R6: restore cross-evidence pooling + reduce contrastive + structured global",
        "experiments": {
            # R6-A: Restore cross-evidence attention (cross_self_mix=0.5 instead of 1.0)
            "R6_cross_self05": {
                "overrides": {**V32_BASE, "model.cross_self_mix": 0.5},
            },
            # R6-B: Full cross-evidence (cross_self_mix=0.0, pure cross attention)
            "R6_cross_only": {
                "overrides": {**V32_BASE, "model.cross_self_mix": 0.0},
            },
            # R6-C: Much lower contrastive weight (0.02 vs 0.1)
            "R6_cw_002": {
                "overrides": {**V32_BASE, "model.proto_contrastive_weight": 0.02},
            },
            # R6-D: No contrastive at all
            "R6_cw_000": {
                "overrides": {**V32_BASE, "model.proto_contrastive_weight": 0.0,
                              "model.use_contrastive_head": False},
            },
            # R6-E: Re-enable structured global
            "R6_struct_global": {
                "overrides": {**V32_BASE, "model.structured_global_enabled": True},
            },
            # R6-F: Lower logit scale (4.0) for softer decisions
            "R6_scale_4": {
                "overrides": {**V32_BASE, "model.prototype_logit_scale_init": 4.0},
            },
        },
    },
    # ── R7: Build on R6_struct_global (best at 94.35%) ────────────────────
    # R6 showed structured_global_enabled is the key lever (+4% over base).
    # cw=0.1 is the sweet spot (0.02/0.0/0.15 all worse). cross_self_mix=1.0 is fine.
    # R7 probes further structural improvements on top of struct_global.
    "R7": {
        "desc": "R7: struct_global + residual, concept_dim, training refinements",
        "experiments": {
            # R7-A: struct_global + higher residual_weight (0.15 vs 0.05)
            "R7_sg_rw015": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.15},
            },
            # R7-B: struct_global + even higher residual_weight (0.3)
            "R7_sg_rw03": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.30},
            },
            # R7-C: struct_global + concept_dim=128, role_dim=32 (richer repr)
            "R7_sg_cd128": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.concept_dim": 128,
                              "model.role_dim": 32},
            },
            # R7-D: struct_global + lower label_smoothing (0.05)
            "R7_sg_ls005": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "task.label_smoothing": 0.05},
            },
            # R7-E: struct_global + balance_weight=0.1 (more balanced prototypes)
            "R7_sg_bal01": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.prototype.balance_weight": 0.1},
            },
            # R7-F: struct_global + wider tau (0.2) for softer prototype routing
            "R7_sg_tau02": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.prototype_temperature": 0.2,
                              "model.prototype_assignment_temperature": 0.2},
            },
        },
    },
    # ── R8: Best config (struct_global + rw=0.30) with extended training ──
    # R7_sg_rw03 achieved 0.9464±0.0091 on T0 in 30ep. R8 tests convergence
    # and fine-tunes around the best config.
    "R8": {
        "desc": "R8: struct_global + rw=0.30 base, convergence and fine-tuning",
        "experiments": {
            # R8-A: Best config, just needs more training
            "R8_best_50ep": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.30},
            },
            # R8-B: rw=0.20 (intermediate)
            "R8_rw02": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20},
            },
            # R8-C: rw=0.40 (slightly higher)
            "R8_rw04": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.40},
            },
            # R8-D: rw=0.50 (aggressive residual)
            "R8_rw05": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.50},
            },
            # R8-E: best + lower weight_decay
            "R8_wd0001": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.30,
                              "task.weight_decay": 0.0001},
            },
            # R8-F: best + cw=0.08 (slightly lower contrastive)
            "R8_cw008": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.30,
                              "model.proto_contrastive_weight": 0.08},
            },
        },
    },
    # ── R9: Deeper architectural exploration ──────────────────────────────
    # R8 showed D0 ~94%, D1 ~94%, D2 ~88% with struct_global+rw=0.20.
    # Need to push D0→95%, D1→99%, D2→97%. R9 explores deeper changes.
    "R9": {
        "desc": "R9: backbone blend, eval augmentation, regularization",
        "experiments": {
            # R9-A: Blend structured+deep (transparent_weight=0.5 instead of 1.0)
            "R9_tw05": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.5},
            },
            # R9-B: eval-time MC augmentation (5 samples)
            "R9_mc5": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.eval_mc_samples": 5,
                              "model.region_ensemble_samples": 1},
            },
            # R9-C: region ensemble during training (3 samples)
            "R9_re3": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.region_ensemble_samples": 3},
            },
            # R9-D: stronger regularization (weight_decay=0.001)
            "R9_wd001": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "task.weight_decay": 0.001},
            },
            # R9-E: fixed model init seed for reproducibility
            "R9_mseed42": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "environment.model_init_seed": 42},
            },
            # R9-F: stability consistency weight to reduce variance
            "R9_scw01": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.stability_consistency_weight": 0.1},
            },
            # R9-G: best + cw=0.05 (even lower contrastive for D2 stability)
            "R9_cw005": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.proto_contrastive_weight": 0.05},
            },
        },
    },
    # ── R10: Feature focus and regularization for D1/D2 ──────────────────
    # v3.3 mainline gets D0~94%, D1~94.5%, D2~91%. Targets: 95/99/97.
    # R10 tests focused features and stronger regularization.
    "R10": {
        "desc": "R10: feature focus, AdamW, lower patches for D1/D2",
        "experiments": {
            # R10-A: v3.3 with fewer patches (32 instead of 64)
            "R10_p32": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_count": 32,
                              "model.freq_band_count": 32},
            },
            # R10-B: v3.3 with wider patches (1024 instead of 512)
            "R10_pw1024": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024},
            },
            # R10-C: v3.3 with AdamW optimizer
            "R10_adamw": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "task.optimizer": "adamw"},
            },
            # R10-D: v3.3 with higher batch size (128)
            "R10_bs128": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "data.batch_size": 128},
            },
            # R10-E: v3.3 with lower lr (0.0003) for stability
            "R10_lr0003": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "task.lr": 0.0003},
            },
            # R10-F: v3.3 with K=2 prototypes + higher logit scale
            "R10_k2_s8": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.num_prototypes_per_class": 2,
                              "model.prototype_logit_scale_init": 8.0},
            },
        },
    },
    # ── R11: Design-level structural improvements ────────────────────────
    # Root cause: transparent_backbone_weight=1.0 ELIMINATES structured global
    # from the main decision path. Structured stats are more domain-invariant.
    # R11 tests proper fusion of structured + deep global evidence.
    "R11": {
        "desc": "R11: design-level fusion - structured+deep blend for domain invariance",
        "experiments": {
            # R11-A: pw1024 + blend structured+deep (tw=0.5)
            "R11_blend05": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.5,
                              "model.time_patch_width": 1024},
            },
            # R11-B: pw1024 + mostly structured (tw=0.3) - more domain-invariant
            "R11_struct03": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.3,
                              "model.time_patch_width": 1024},
            },
            # R11-C: pw1024 + blend + restore cross-attention (self_mix=0.0)
            "R11_blend_cross": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.5,
                              "model.cross_self_mix": 0.0,
                              "model.time_patch_width": 1024},
            },
            # R11-D: pw1024 + blend + softer cross-pool (tau=1.0)
            "R11_blend_soft": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.5,
                              "model.cross_pool_tau": 1.0,
                              "model.time_patch_width": 1024},
            },
            # R11-E: pw1024 + blend + higher contrastive (0.15)
            "R11_blend_cw15": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.transparent_backbone_weight": 0.5,
                              "model.proto_contrastive_weight": 0.15,
                              "model.time_patch_width": 1024},
            },
        },
    },
    # ── R12: Cooperative prototypes and multi-path evidence ───────────────
    # D1 gap requires code-level design change. Test cooperative prototypes
    # (separate time/freq/joint heads) and evidence diversity approaches.
    "R12": {
        "desc": "R12: cooperative prototypes + evidence diversity for D1",
        "experiments": {
            # R12-A: v3.4 + cooperative prototypes (time/freq/joint heads)
            "R12_coop": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.cooperative_prototypes_enabled": True},
            },
            # R12-B: v3.4 (pw512) + cooperative prototypes (better for D1)
            "R12_coop_512": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.cooperative_prototypes_enabled": True},
            },
            # R12-C: v3.4 + prototype diversity weight (prevent collapse)
            "R12_div01": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.prototype_diversity_weight": 0.1},
            },
            # R12-D: v3.4 + occupancy weight (encourage full prototype usage)
            "R12_occ01": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.prototype_occupancy_weight": 0.1},
            },
            # R12-E: v3.4 + adaptive effective K (auto-prune dead prototypes)
            "R12_adpk": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.adaptive_effective_k_enabled": True},
            },
        },
    },
    "R13": {
        "desc": "R13: multi-granularity local evidence (stochastic training + logit ensemble)",
        "experiments": {
            # R13-A: pw1024 primary + pw512 secondary multi-granularity
            "R13_mg_1024_512": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.time_patch_widths": [512]},
            },
            # R13-B: pw512 primary + pw1024 secondary (reverse direction)
            "R13_mg_512_1024": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 512,
                              "model.time_patch_widths": [1024]},
            },
            # R13-C: pw1024 primary + pw256 + pw512 (three scales)
            "R13_mg_3scale": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.time_patch_widths": [256, 512]},
            },
            # R13-D: pw1024 primary + pw512 secondary with pw512 as primary
            #  (ensures h_raw stats come from pw512 which is better for D1)
            "R13_mg_512_pri": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 512,
                              "model.time_patch_widths": [1024],
                              "model.num_prototypes_per_class": 3},
            },
        },
    },
    "R14": {
        "desc": "R14: cross-patch instance normalization (domain-invariant evidence)",
        "experiments": {
            # R14-A: v3.4 pw1024 + cross-patch norm
            "R14_cpn_1024": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.cross_patch_norm_enabled": True},
            },
            # R14-B: v3.4 pw512 + cross-patch norm (pw512 was best for D1)
            "R14_cpn_512": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.cross_patch_norm_enabled": True},
            },
        },
    },
    # ── R15: Structured Evidence Dropout ─────────────────────────────────
    # Randomly zero out entire patches/bands during training so the model
    # learns to diagnose from partial evidence.  Combined with softer
    # prototype temperature for more robust matching.
    "R15": {
        "desc": "R15: structured evidence dropout for domain robustness",
        "experiments": {
            # R15-A: evidence_dropout=0.15 on v3.4 pw1024
            "R15_ed015": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.evidence_dropout_rate": 0.15},
            },
            # R15-B: evidence_dropout=0.25 on v3.4 pw1024
            "R15_ed025": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.evidence_dropout_rate": 0.25},
            },
            # R15-C: evidence_dropout=0.15 + softer temperature (0.25)
            "R15_ed015_tau025": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.time_patch_width": 1024,
                              "model.evidence_dropout_rate": 0.15,
                              "model.prototype_temperature": 0.25,
                              "model.prototype_assignment_temperature": 0.25},
            },
            # R15-D: evidence_dropout=0.15 on pw512 (pw512 was better for D1)
            "R15_ed015_pw512": {
                "overrides": {**V32_BASE,
                              "model.structured_global_enabled": True,
                              "model.classifier.residual_weight": 0.20,
                              "model.evidence_dropout_rate": 0.15},
            },
        },
    },
}


def deep_set(d: dict, key: str, value):
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def make_config(overrides: dict, domain: int, episodes: int, epochs: int,
                patience: int | None = None) -> dict:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        deep_set(cfg, k, v)
    # Leave-one-out: target=domain, source=remaining domains
    all_domains = [0, 1, 2]
    cfg["task"]["target_domain_id"] = [domain]
    cfg["task"]["source_domain_id"] = [d for d in all_domains if d != domain]
    cfg["environment"]["iterations"] = episodes
    cfg["trainer"]["num_epochs"] = epochs
    cfg["trainer"]["patience"] = patience if patience is not None else epochs
    return cfg


def collect_results(output_dir: str) -> dict:
    results = {"iterations": []}
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            if fname.startswith("test_result_") and fname.endswith(".csv"):
                try:
                    with open(os.path.join(root, fname)) as f:
                        reader = csv_mod.DictReader(f)
                        for row in reader:
                            acc = float(row.get("test_acc_RM_023_HIT23",
                                                 row.get("test_acc", 0)))
                            results["iterations"].append({
                                "test_acc": acc,
                                "test_proto_nce": float(row.get("test_proto_nce", 0)),
                                "test_total_loss": float(row.get("test_total_loss", 0)),
                                "decision_anchor_acc": float(row.get("test_decision_anchor_acc", 0)),
                            })
                except Exception:
                    pass
    accs = [r["test_acc"] for r in results["iterations"] if r.get("test_acc")]
    if accs:
        import numpy as np
        results["mean_acc"] = float(np.mean(accs))
        results["std_acc"] = float(np.std(accs))
        results["min_acc"] = float(np.min(accs))
        results["max_acc"] = float(np.max(accs))
        results["per_iter"] = accs
    return results


def run_one(exp_name: str, exp: dict, domain: int,
            episodes: int, epochs: int, base_output: str,
            patience: int | None = None) -> dict:
    cfg = make_config(exp["overrides"], domain, episodes, epochs, patience)
    tag = f"{exp_name}_d{domain}"
    exp_dir = os.path.join(base_output, tag)
    os.makedirs(exp_dir, exist_ok=True)
    cfg["environment"]["output_dir"] = exp_dir

    cfg_path = os.path.join(exp_dir, "_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    from types import SimpleNamespace
    from src.Pipeline_01_default import pipeline

    args = SimpleNamespace(
        config=cfg_path, config_path=cfg_path,
        notes=f"{tag}", override=None, local_config=None,
    )
    t0 = time.time()
    try:
        pipeline(args)
    except Exception as e:
        traceback.print_exc()
        return {"tag": tag, "error": str(e), "duration_s": time.time() - t0}
    dur = time.time() - t0
    res = collect_results(exp_dir)
    res["tag"] = tag
    res["domain"] = domain
    res["duration_s"] = dur
    target = ACC_TARGETS[domain]
    if "mean_acc" in res:
        status = "PASS" if res["min_acc"] >= target else "FAIL"
        res["status"] = status
        print(f"  >> {tag}: acc={res['mean_acc']:.4f} ± {res['std_acc']:.4f} "
              f"[{res['min_acc']:.4f}, {res['max_acc']:.4f}] "
              f"target={target:.0%} {status}  ({dur:.0f}s)")
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, choices=list(ROUNDS.keys()))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--output-dir", type=str,
                    default="results/demo/tf_multi_proto_dg_multidomain")
    ap.add_argument("--domains", nargs="*", type=int, default=None)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="Early stopping patience (default: same as epochs)")
    args = ap.parse_args()

    round_cfg = ROUNDS[args.round]
    domains = args.domains or TARGET_DOMAINS
    exp_names = args.experiments or list(round_cfg["experiments"].keys())

    print(f"\n{'='*70}")
    print(f"  Round: {args.round} — {round_cfg['desc']}")
    print(f"  Domains: {domains}, Episodes: {args.episodes}, Epochs: {args.epochs}, Patience: {args.patience or args.epochs}")
    print(f"  Targets: {', '.join(f'D{d}>={ACC_TARGETS[d]:.0%}' for d in domains)}")
    print(f"{'='*70}")

    all_res = []
    for ename in exp_names:
        if ename not in round_cfg["experiments"]:
            print(f"WARNING: unknown experiment '{ename}'")
            continue
        exp = round_cfg["experiments"][ename]
        for domain in domains:
            r = run_one(ename, exp, domain, args.episodes, args.epochs,
                        args.output_dir, args.patience)
            all_res.append(r)

    # ── Summary ──
    print(f"\n\n{'='*80}")
    print(f"  ROUND {args.round.upper()} SUMMARY")
    print(f"{'='*80}")
    hdr = f"{'Experiment':<35} {'Dom':>3} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'Tgt':>5} {'Status':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_res:
        if "mean_acc" in r:
            tgt = ACC_TARGETS.get(r["domain"], 0)
            st = r.get("status", "?")
            print(f"{r['tag']:<35} D{r['domain']:>1} {r['mean_acc']:>6.4f} "
                  f"{r['std_acc']:>6.4f} {r['min_acc']:>6.4f} {r['max_acc']:>6.4f} "
                  f"{tgt:>5.2f} {st:>5}")
        else:
            print(f"{r.get('tag','?'):<35} ERROR: {r.get('error','?')[:30]}")

    # Per-domain best
    print(f"\n  Per-domain best:")
    for d in domains:
        best = max((r for r in all_res if r.get("domain") == d and "mean_acc" in r),
                   key=lambda r: r["mean_acc"], default=None)
        if best:
            tgt = ACC_TARGETS[d]
            st = "PASS" if best["min_acc"] >= tgt else "FAIL"
            print(f"    D{d}: best={best['mean_acc']:.4f} ({best['tag']}) "
                  f"target={tgt:.0%} {st}")

    summary_path = os.path.join(args.output_dir, f"_summary_{args.round}.json")
    with open(summary_path, "w") as f:
        json.dump(all_res, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
