# TSPN Physics-Conditioned Contrastive Workflow

The original multi-branch (`support_support`, `support_query`, `query_query`) ablation suite has been retired. All legacy configs and scripts now live under `archive/legacy_tspn_contrastive_ablation/` for reference-only use. The active workflow replaces branch toggles with a single physics-conditioned contrastive (PCC) objective driven by:

1. **Convex Combination Projector** – mixes backbone operators via simplex-constrained matrix `A`.
2. **SPD Coupling Metric** – applies `M = D + L L^T` and learnable temperature vector `τ` (with `‖L‖²_F` regularisation).
3. **PCC Loss** – builds positives from cross-domain supports + physics-consistent query views and negatives from inter-class samples, all under the SPD kernel.

## Running the New Pipeline

1. Start from `configs/demo/X_Single_DG/TSPN_FewShot/shared.yaml`. The `model.contrastive` section now exposes:
   - `physical_projector.indicator_dim`, `simplex_eps`, `spd_rank`, `tau_*`.
   - `reg_weight` controlling `λ_reg‖L‖²_F`.
2. Enable PCC by importing `configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml` (or the scenario-specific variants like `JUST.yaml`, `ottawa.yaml`). Each file specifies:
   - `task.contrastive.loss_weight` → `λ_pcc`.
   - `task.contrastive.reg_weight` → projector regulariser weight.
   - `task.contrastive.pcc` → positive/negative weighting, cosine filter threshold, and physics augmentation knobs.
3. Launch experiments with the standard entrypoint, e.g.:

```bash
python main.py --config configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml \
  --pipeline Pipeline_02_pretrain_fewshot
```

### Episodic Guidance

- The sampler continues to use class/domain selections declared in `task.few_shot` but all contrastive supervision now occurs within a single PCC loss.
- Negatives come exclusively from inter-class support/query samples until physics-violating augmentations are re-enabled.

## Legacy References

- Legacy configs/shell helpers: `archive/legacy_tspn_contrastive_ablation/`
- Former documentation of ss/sq/qq branches: `docs/tspn_loss_composition.md` (see “Legacy Branches” appendix).

These assets are preserved solely for historical comparison and should not be used for new experiments.
