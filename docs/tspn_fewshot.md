# TSPN Few-Shot & Contrastive Domain Generalisation

This document outlines how to run the newly added few-shot, contrastive, and explainability workflows for the TSPN baseline.

## Configuration presets

The new presets live under `configs/demo/X_Single_DG/TSPN_FewShot/`.

- `base.yaml` keeps the original HUST workflow while adding disabled toggles for few-shot sampling, contrastive heads, and explainability.
- `contrastive.yaml` enables the `TSPNContrastive` model wrapper, the physical indicator projector, and the few-shot sampler for contrastive DG experiments. It defaults `task.few_shot.format` to `"episode"` so support/query semantics persist through training.
- `explainable.yaml` extends the contrastive preset with explainability exports that save domain-level feature summaries and indicator-to-class weights.

Each file documents when to switch `few_shot.enabled`, `model.contrastive.enabled`, and `task.explainability.enabled`. When all toggles remain `false`, the run behaves identically to the original `configs/demo/X_Single_DG/TSPN/HUST.yaml` benchmark.

### Configuring the physical indicator space

- The `model.contrastive.physical_projector` block controls the interpretable indicator layer. Set `enabled=true` and `indicator_dim` to the desired number of indicators. Optional `sparsity_l1`/`sparsity_group` penalties regularise the projection weights, while the `prior` section can load a mask (e.g. `.npy`) to enforce known physics.
- When the projector is active, `projection_dim=0` (or matching `indicator_dim`) keeps the contrastive head in the same physical space. Set `projection_hidden=0` to disable extra non-linear layers for maximum interpretability.
- `self.network.explain_indicator_weights()` exposes the per-class indicator weights, and validation/test logs now include `*_indicator_weight_norm` plus `*_indicator_penalty_total` if sparsity penalties are active.

### Support-aware supervision and adaptive weighting

- `task.contrastive.support_loss` defines how support samples contribute gradients. Choose `"cross_entropy"`, `"prototype"`, or `"hybrid"` and scale components with `ce_weight`, `prototype_weight`, and the global `loss_weight`.
- `task.contrastive.weighting` selects the contrastive branch balancer: `"gradnorm"` adjusts weights based on loss magnitudes, while `"uncertainty"` uses running variance estimates. Weights and per-branch losses are logged as `*_contrastive_{branch}_weight` / `*_loss`.
- For legacy behaviour, leave `support_loss.loss_weight=0` and `weighting.strategy="uniform"`.

### Loss composition reference

A detailed breakdown of the combined episode loss (support CE, prototype alignment, branch-specific InfoNCE, and projector penalties) is available in [`docs/tspn_loss_composition.md`](tspn_loss_composition.md).

## Running an experiment

```bash
python main.py --config configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml
```

The config can also be combined with command-line overrides, for example to adjust the number of support/query shots:

```bash
python main.py --config configs/demo/X_Single_DG/TSPN_FewShot/contrastive.yaml \
    --set task.few_shot.support_per_class=3 --set task.few_shot.query_per_class=6
```

Set `--set task.few_shot.format=episode` to opt into the structured episode pipeline when using other presets, or `--set task.few_shot.format=flat` to fall back to legacy batching.

## Episode-aware workflow

The new [`docs/tspn_episode_pipeline.md`](tspn_episode_pipeline.md) guide documents the full path from episode sampling to gradient updates, including prototype construction, contrastive loss composition, and artefact exports.

Explainability artefacts are written under `<output_dir>/<run-name>/explainability/<stage>/`. Each domain receives a CSV of the top features ranked by mean absolute activation and a label histogram for quick inspection. Set `task.explainability.summary.save_embeddings=true` to persist the raw projected embeddings (`.pt` files) for custom analysis.

```mermaid
flowchart LR
    A["原始数据\n(IdIncludedDataset)"] --> B["元数据加载"]
    B --> C["FewShotDGSampler\n(支持/查询划分)"]
    C --> D["EpisodeCollate\n(EpisodeBatch)"]

    D --> E["支持前向\nTSPNContrastive"]
    D --> F["查询前向\nTSPNContrastive"]
    E --> G["支持嵌入 / Logits"]
    F --> H["查询嵌入 / Logits"]
    H --> I["Projection Head\nInfoNCE 投影"]

    G --> J["支持监督\nCE + Prototype"]
    H --> K["查询交叉熵"]
    I --> L1["support_support"]
    I --> L2["support_query"]
    I --> L3["query_query"]

    L1 & L2 & L3 --> M["分支权重调度"]
    M --> N["Contrastive Loss"]

    J & K & N --> O["总损失"]
'''
