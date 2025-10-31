# TSPN Few-Shot Loss Composition

This note spells out the objective optimised by the interpretable `TSPNContrastive` pipeline and how each term is controlled in configuration files.

## Total objective

For an episode with query set `Q`, support set `S`, projector parameters `W`, and branch weights `w_b`, the total loss is

$$
\mathcal{L}_{\text{total}} =
\underbrace{\mathcal{L}_{\text{CE}}(Q)}_{\text{query classification}}
+ \lambda_{\text{sup}}\!\left[
  \alpha_{\text{ce}}\mathcal{L}_{\text{CE}}(S) +
  \alpha_{\text{proto}}\mathcal{L}_{\text{proto}}(S)
\right]
+ \lambda_{\text{proj}}\mathcal{R}_{\text{proj}}(W)
+ \lambda_{\text{ctr}}\sum_{b \in \{\text{ss}, \text{sq}, \text{qq}\}} w_b\,\mathcal{L}_{b},
$$

where

- `ss` / `sq` / `qq` denote the support–support, support–query, and query–query InfoNCE branches.
- `λ_sup = task.contrastive.support_loss.loss_weight`.
- `α_ce`, `α_proto` come from `ce_weight` / `prototype_weight`.
- `λ_proj` is implicit in the projector penalties (e.g. `sparsity_l1`).
- `λ_ctr = task.contrastive.loss_weight`.

All scalar hyperparameters map directly to YAML entries and can be set to zero to recover the legacy behaviour.

## Component details

### Query cross-entropy

`task._compute_loss` is applied to query logits vs. labels. This is always active and reported as `{stage}_loss` plus `{stage}_{domain}_loss`.

### Support supervision

Configured under `task.contrastive.support_loss`:

- `mode`: `"cross_entropy"`, `"prototype"`, or `"hybrid"`.
- `loss_weight` (`λ_sup`): global scaling applied after combining the CE and prototype sub-losses.
- `ce_weight`, `prototype_weight` (`α_ce`, `α_proto`): relative weighting before the global factor.
- `prototype_metric`: `"cosine"` minimises `(1 - cos θ)`; `"euclidean"` minimises MSE to the class prototype.

Metrics:
- `{stage}_support_ce`, `{stage}_support_proto` — raw components before weighting.
- `{stage}_support_ce_weighted`, `{stage}_support_proto_weighted` — post component weights.
- `{stage}_support_loss`, `{stage}_support_weighted_loss` — combined totals before/after `loss_weight`.

### Projector regularisation

When `model.contrastive.physical_projector.enabled=true`, penalties returned by the projector are added automatically. Available terms:

- `projector_l1` — L1 sparsity scaled by `sparsity_l1`.
- `projector_group` — row L2 norms scaled by `sparsity_group`.
- `projector_max_active` — surplus weight mass beyond `max_active` (if specified).

Metrics:
- `{stage}_projector_l1`, `{stage}_projector_group`, `{stage}_projector_max_active`.
- `{stage}_indicator_penalty_total` — sum of the active penalties.

### Contrastive branches

Let $\mathcal{A}_b$ be the anchors and $\mathcal{C}_b$ the candidate pool for branch $b$. Given anchor features $z_i$ and candidate features $z_j$ (L2-normalised when `normalize_embeddings=true`) and temperature $\tau$, their similarity is

$$
s_{ij} = \frac{z_i^\top z_j}{\tau}.
$$

Anchors are only kept if they have at least one positive candidate:

$$
P_i =
\begin{cases}
\{j \in \mathcal{C}_b \mid y_j = y_i\}, & \text{mode}=\text{``supervised''} \\[4pt]
\{j \in \mathcal{C}_b \mid y_j = y_i \land \text{domain}_j \neq \text{domain}_i\}, & \text{mode}=\text{``domain-aware''}
\end{cases},
\qquad
\mathcal{A}_b^\star = \{ i \in \mathcal{A}_b \mid |P_i| > 0 \}.
$$

Self-positives ($i=j$) are excluded when $\mathcal{A}_b$ and $\mathcal{C}_b$ refer to the same tensor. For every valid anchor $i \in \mathcal{A}_b^\star$, the InfoNCE contribution is

$$
\ell_i = -\frac{1}{|P_i|}\sum_{j\in P_i} \log \frac{\exp(s_{ij})}{\sum_{k\in \mathcal{C}_b} \exp(s_{ik})}.
$$

The branch loss then averages across the valid anchors:

$$
\mathcal{L}_b = \frac{1}{|\mathcal{A}_b^\star|} \sum_{i\in \mathcal{A}_b^\star} \ell_i.
$$

By default the task runs in `mode="supervised"`, so positives are selected purely by matching labels. Switching to `mode="domain-aware"` additionally requires positives to come from different domains.

Branch definitions:

- `query_query` (`qq`): $\mathcal{A}_{qq} = \mathcal{C}_{qq} = Q$ with self-pairs masked.
- `support_support` (`ss`): $\mathcal{A}_{ss} = \mathcal{C}_{ss} = S$ with self-pairs masked.
- `support_query` (`sq`): bidirectional average of support→query and query→support InfoNCE terms,
  $$
  \mathcal{L}_{sq} = \tfrac{1}{2}\left(\mathcal{L}_{S\to Q} + \mathcal{L}_{Q\to S}\right),
  $$
  where each directional loss follows the definition above with its own anchor set.

Branch weights $w_b$ are computed in `_combine_contrastive_branches` and rescale the raw losses before $\lambda_{\text{ctr}}$ is applied:

- `strategy="uniform"`/`"none"` — proportional to positive counts (default).
- `strategy="gradnorm"` — GradNorm-style ratios based on current loss vs. EMA (`alpha`).
- `strategy="uncertainty"` — inverse variance of branch losses estimated via EMA.
- Otherwise — base weights raised to the power `alpha`.

Metrics:
- `{stage}_contrastive_{branch}_loss`, `{stage}_contrastive_{branch}_positives`, `{stage}_contrastive_{branch}_anchors`.
- `{stage}_contrastive_{branch}_weight` — adaptive weights.
- `{stage}_contrastive_loss`, `{stage}_contrastive_weighted_loss` — combined loss before/after `λ_ctr`.

See `branch_weights_batches.csv` and `branch_weight_summary.csv` under the explainability output directory for per-batch diagnostics.

### Explainability artefacts

When `task.explainability.enabled=true`, the exporter stores:

- `indicator_weights.csv` / `.pt` — per-class indicator weights (and optional bias).
- `projection_matrix.pt` — the physical projector matrix when `physical_projector` is active.
- Top-feature CSVs, label histograms, and optional embedding tensors (controlled by `summary.save_embeddings`).

These files make it easy to trace predictions back to physically meaningful indicators and to audit the adaptive weighting behaviour across episodes.
