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

1. **Support encoding** (no gradient):
   - Support tensors are encoded via `network.encode` under `torch.no_grad()`.
   - Label-wise prototypes are computed by averaging embeddings within each support slice.
   - Support projections are detached and reused as positive anchors for contrastive training.
2. **Query forward** (with gradient):
   - Query tensors are forwarded through the contrastive TSPN wrapper with `return_embeddings=True`.
   - Classification logits and projected embeddings are produced only for queries.
3. **Loss evaluation**:
   - Cross-entropy is computed on query logits using their ground-truth labels.
   - Optional regularisation losses reuse the existing factory utilities.
   - The contrastive objective concatenates detached support projections with query projections so queries receive gradients while supports contribute positives without gradient flow.
4. **Backpropagation**:
   - Gradients trickle back exclusively through the query branch (logits and projections) ensuring the support set acts as a static context.
   - Optimiser updates mirror the legacy trainer; only the loss composition changes.

## Loss Composition

| Component | Source | Notes |
|-----------|--------|-------|
| `ce_loss` | Query logits vs. query labels | Primary classification signal |
| Regularisation | Configured in `task.regularization` | Applied unchanged |
| Contrastive | Query projections (gradients) + detached support/query positives | Weighted by `task.contrastive.loss_weight` |

Total loss = `ce_loss + regularisation + contrastive_weight * contrastive_loss`.

## Outputs and Artefacts

- **Metrics**: All accuracy/loss metrics now reflect query batch sizes; the logged batch size equals the number of query samples.
- **Prototype norms**: When `task.contrastive.log_components=true`, average prototype norms are emitted per stage (`train_prototype_norm_mean`, etc.).
- **Explainability**: Validation/test embeddings cache query projections and file IDs, so produced attribution files correspond to the evaluation portion of each episode.
- **Saved embeddings**: If `task.explainability.summary.save_embeddings=true`, outputs in `save/<run>/explainability/` contain query embeddings and the detached support prototypes for post-hoc analysis.

## Configuration Checklist

1. Set `task.few_shot.enabled=true` and `task.few_shot.format="episode"` to activate episode-aware mode.
2. Tune `support_per_class` / `query_per_class`; shortfalls are logged but layouts still preserve per-class slices.
3. Adjust `task.contrastive.loss_weight` and `task.contrastive.mode` to control the influence of support-derived positives.
4. Leave `task.few_shot.format` unset or `"flat"` to maintain prior behaviour.

## Verifying the Setup

- **Dry run**: `python main.py --config configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml --pipeline Pipeline_01_default --set environment.iterations=1 --set trainer.num_epochs=1`
- **Key logs**: Look for `train_batch_size`, `*_contrastive_loss`, and `*_prototype_norm_mean` entries to confirm episode-aware batching is active.
- **Artifacts**: Inspect `results/<run>/explainability/` for per-episode embeddings reflecting the query-only gradient path.

With these changes, the support/query semantics selected during sampling are preserved through training, enabling stable few-shot contrastive episodes and clearer separation between adaptation and evaluation roles.
