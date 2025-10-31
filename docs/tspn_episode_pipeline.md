# TSPN Episode-Aware Few-Shot Pipeline

This document explains how the episode-aware setting propagates support/query semantics from data ingestion to gradient updates when `task.few_shot.format="episode"` is enabled.

## Overview

1. **Sampler stage** – `FewShotDGSampler` now records, for every episode, the ordered support/query indices, class identifiers, and domain metadata.
2. **Collate stage** – `EpisodeCollate` reshapes each batch into support/query tensors and exposes an `EpisodeBatch` dataclass consumed by the task.
3. **Task stage** – `contrastive_classification` encodes support samples to build prototypes/positive queues, then forwards query samples through the network, using query losses to drive gradients.
4. **Outputs** – Contrastive diagnostics, prototype summaries, and optional explainability artefacts are logged with the new semantics.

## Data Flow

| Stage | Input | Transformation | Output |
|-------|-------|----------------|--------|
| Metadata load | `metadata_*.xlsx` | `MetadataAccessor` | metadata rows per `file_id` |
| Dataset | `IdIncludedDataset` | Windowing & normalisation | dict `{"x": window, "y": label, "file_id": id}` |
| Sampler | dataset indices | Episode layout + index list | ordered support/query indices + `EpisodeLayout` |
| Collate | sample dictionaries | tensor stack + label grouping | `EpisodeBatch` (support/query tensors, metadata) |
| Task | `EpisodeBatch` | encode support, forward query, compute losses | metrics, gradients, artefacts |

## Episode Sampling Details

- `FewShotDGSampler` preserves the original behaviour (yielding a flat index list) while pushing an `EpisodeLayout` object into a queue.
- Each `EpisodeLayout` entry stores:
  - `system_id`, `domain_id`, `label_id`
  - support/query start indices and counts
  - requested versus effective shot counts (for shortfall logging)
- Downstream consumers retrieve layouts with `sampler.pop_layout()`.

## Collation to EpisodeBatch

`EpisodeCollate` performs the following steps for each episode:

1. Pop the next `EpisodeLayout` from the sampler queue.
2. Stack feature tensors and labels in the original order via `default_collate`.
3. Split the stacked tensors into support/query blocks according to the recorded spans.
4. Build `EpisodeLabelView` entries containing per-class slices, local label ids, and original metadata.
5. Return an `EpisodeBatch` dataclass exposing:
   - `support_x`, `support_y`, `support_file_ids`
   - `query_x`, `query_y`, `query_file_ids`
   - `label_views` (slice definitions) and the raw `EpisodeLayout`
   - `flat_batch` – the original collated dictionary for compatibility fallbacks

When episode format is disabled (default), standard PyTorch collation is used.

## Forward & Backward Pass (Episode Mode)

1. **Support encoding & logits**:
   - Support tensors are forwarded through `TSPNContrastive` with gradients enabled, producing indicator embeddings, logits, and projector penalties.
   - Label-wise prototypes are computed by averaging support embeddings within each slice and can be used for prototype alignment loss (`task.contrastive.support_loss.mode="prototype"` or `"hybrid"`).
   - Support logits contribute to cross-entropy (`mode="cross_entropy"`/`"hybrid"`) so labelled shots actively steer the backbone.
2. **Query forward** (with gradient):
   - Query tensors are forwarded through the contrastive TSPN wrapper with `return_embeddings=True`.
   - Classification logits and projected embeddings are produced only for queries.
3. **Loss evaluation**:
   - Query CE and existing regularisation losses mirror the legacy pipeline.
   - Support CE/prototype terms are aggregated according to `task.contrastive.support_loss` weights.
   - Contrastive similarity matrices are split into `support_support`, `support_query`, and `query_query` branches. Positive counts normalise each branch before adaptive re-weighting (`task.contrastive.weighting.strategy`).
   - Physical projector penalties (e.g. L1 sparsity) are added automatically when `model.contrastive.physical_projector.enabled=true`.
4. **Backpropagation**:
   - Gradients propagate through both support and query embeddings whenever support losses are active; branch weights balance the InfoNCE contributions.
   - Logged metrics expose per-branch losses/weights and projector penalties to simplify debugging.

## Loss Composition

| Component | Source | Notes |
|-----------|--------|-------|
| `query_ce` | Query logits vs. query labels | Primary classification signal |
| `support_ce`, `support_proto` | Configured via `task.contrastive.support_loss` | Weighted by component weights then scaled by `loss_weight` |
| Projector penalties | `model.contrastive.physical_projector` sparsity/prior terms | Logged as `*_indicator_penalty_*` |
| Regularisation | Existing `task.regularization` block | Applied unchanged |
| Branch contrastive | `support_support`, `support_query`, `query_query` InfoNCE branches | Normalised by positive counts then re-weighted (GradNorm/uncertainty) before multiplying by `task.contrastive.loss_weight` |

See [`docs/tspn_loss_composition.md`](tspn_loss_composition.md) for a derivation of the full objective.

## Outputs and Artefacts

- **Metrics**: Query-aligned accuracy, support loss components, branch losses/weights (`*_contrastive_{branch}_*`), and projector penalties are logged per stage.
- **Prototype norms**: When `task.contrastive.log_components=true`, average prototype norms are emitted per stage (`train_prototype_norm_mean`, etc.).
- **Explainability**: Validation/test caches now include indicator weights, optional projection matrices, and batch-wise branch weights. Exports provide:
  - `indicator_weights.csv` / `.pt` – per-class indicator mappings.
  - `branch_weights_batches.csv` / `branch_weight_summary.csv` – diagnostics for adaptive weighting.
  - Existing top-feature CSVs and optional embedding dumps (`task.explainability.summary.save_embeddings=true`).

## Configuration Checklist

1. Set `task.few_shot.enabled=true` and `task.few_shot.format="episode"` to activate episode-aware mode.
2. Configure `model.contrastive.physical_projector` (indicator dimension, sparsity, optional prior) to expose the interpretable embedding space.
3. Tune `task.contrastive.support_loss` and `task.contrastive.weighting` to balance supervised signals and InfoNCE branches; set `loss_weight=0` or `strategy="uniform"` for legacy behaviour.
4. Adjust `support_per_class` / `query_per_class`; shortfalls are logged but layouts still preserve per-class slices.
5. Leave `task.few_shot.format` unset or `"flat"` to maintain prior behaviour.

## Verifying the Setup

- **Dry run**: `python main.py --config configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml --pipeline Pipeline_01_default --set environment.iterations=1 --set trainer.num_epochs=1`
- **Key logs**: Expect `*_indicator_penalty_total`, `*_support_ce/proto`, and `*_contrastive_{branch}_weight` alongside `train_batch_size` to confirm projector and adaptive weighting are active.
- **Artifacts**: Inspect `results/<run>/explainability/` for indicator weight CSV/pt files, branch weight summaries, and per-episode embeddings.

With these changes, the support/query semantics selected during sampling are preserved through training, enabling stable few-shot contrastive episodes and clearer separation between adaptation and evaluation roles.
