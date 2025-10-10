# TSPN Few-Shot & Contrastive Domain Generalisation

This document outlines how to run the newly added few-shot, contrastive, and explainability workflows for the TSPN baseline.

## Configuration presets

The new presets live under `configs/demo/X_Single_DG/TSPN_FewShot/`.

- `base.yaml` keeps the original HUST workflow while adding disabled toggles for few-shot sampling, contrastive heads, and explainability.
- `contrastive.yaml` enables the `TSPNContrastive` model wrapper and few-shot sampler for contrastive DG experiments. It defaults `task.few_shot.format` to `"episode"` so support/query semantics persist through training.
- `explainable.yaml` extends the contrastive preset with explainability exports that save domain-level feature summaries.

Each file documents when to switch `few_shot.enabled`, `model.contrastive.enabled`, and `task.explainability.enabled`. When all toggles remain `false`, the run behaves identically to the original `configs/demo/X_Single_DG/TSPN/HUST.yaml` benchmark.

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
