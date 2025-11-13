# TSPN Few-Shot Loss Composition (Physics-Conditioned)

The refreshed `TSPNContrastive` pipeline optimises a single physics-conditioned contrastive (PCC) loss alongside query cross-entropy and the SPD regulariser. Legacy branch-specific descriptions are retained in `archive/legacy_tspn_contrastive_ablation/` for reference.

## Total Objective

For an episodic batch with query set `Q`, support set `S`, projector parameters `(A, D, L, τ)`, the loss is:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}(Q)
+ \lambda_{\mathrm{pcc}} \mathcal{L}_{\mathrm{PCC}}(Q; S)
+ \lambda_{\mathrm{reg}} \lVert L \rVert_F^2.
$$

- `λ_pcc = task.contrastive.loss_weight`
- `λ_reg = task.contrastive.reg_weight`
- `‖L‖²_F` reported via `{stage}_metric_reg` and `{stage}_metric_reg_weighted`

### Physics-Constrained Projector

`model.contrastive.physical_projector` mixes backbone operators via a simplex-constrained matrix `A`:

- `indicator_dim`: number of convex axes (default 64)
- `simplex_eps`: diagnostic tolerance for column sums
- `spd_rank`: low-rank factor size for `L`
- `tau_init`, `tau_min`: initialise the learnable temperature vector

Diagnostics logged each step:
- `{stage}_simplex_max_deviation`
- `{stage}_simplex_min_value`

### Query Cross-Entropy

Identical to the legacy pipeline: query logits are supervised with `task._compute_loss`. Metrics:
- `{stage}_loss`, `{stage}_{dataset}_loss`

### PCC Construction

Configured under `task.contrastive.pcc`:

- `lambda_pos`, `lambda_neg`: weights inside `w⁺ = exp(-λ_pos R_phys)` / `w⁻ = exp(+λ_neg R_phys)`
- `cos_threshold_deg`: false-negative filter comparing negatives against class prototypes via the SPD cosine
- `augmentation`: physics-consistent query views (`T_φ`) created via time shift, amplitude scaling, stretch, and ripples

Negatives currently exclude physics-violating augmentations (`T_ψ`) per user request. The builder automatically raises descriptive errors when anchors lack positives or negatives so sampler settings can be adjusted.

The PCC loss is reported through:
- `{stage}_pcc_loss`
- `{stage}_pcc_positive_count`
- `{stage}_pcc_anchor_count`
- `{stage}_pcc_weighted_loss = λ_pcc · ℒ_PCC`

### Explainability Hooks

The previous post-hoc export path (`export_tspn_explainability`) is temporarily disabled while we investigate the logging bug. Runtime metrics still expose simplex diagnostics, indicator penalties, and SPD states for live dashboards, but no artifacts are written after each epoch until the issue is resolved.

## Legacy Branches

The ss/sq/qq formulations, GradNorm-based weighting, and support-alignment parameters described in earlier revisions are preserved in the archived documentation but are no longer part of the active training objective.
