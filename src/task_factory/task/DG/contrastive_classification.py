"""Domain generalisation classification task with contrastive learning."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F

from ...Default_task import Default_task
from ....utils.explainability import export_tspn_explainability


class task(Default_task):
    """Extends the default DG classification task with a contrastive objective."""

    def __init__(
        self,
        network,
        args_data,
        args_model,
        args_task,
        args_trainer,
        args_environment,
        metadata,
    ):
        super().__init__(
            network=network,
            args_data=args_data,
            args_model=args_model,
            args_task=args_task,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=metadata,
        )

        task_cfg = getattr(self.args_task, "contrastive", None)
        model_cfg = getattr(self.args_model, "contrastive", None)

        def _get(cfg, key, default):
            return getattr(cfg, key, default) if cfg and hasattr(cfg, key) else default

        self.contrastive_weight = float(
            _get(task_cfg, "loss_weight", _get(model_cfg, "loss_weight", 0.0))
        )
        self.temperature = float(
            max(_get(task_cfg, "temperature", _get(model_cfg, "temperature", 1.0)), 1e-6)
        )
        self.normalize_embeddings = bool(_get(task_cfg, "normalize_embeddings", True))
        self.log_components = bool(_get(task_cfg, "log_components", False))
        self.mode = _get(task_cfg, "mode", _get(model_cfg, "mode", "supervised"))

        self._explainability_cfg = getattr(self.args_task, "explainability", None)
        self._explainability_enabled = bool(
            getattr(self._explainability_cfg, "enabled", False)
        )
        self._cached_embeddings: Dict[str, list] = {"val": [], "test": []}

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        file_id = batch.get("file_id")
        task_id = batch.get("task_id")
        return self.network(x, file_id, task_id, return_embeddings=True)

    def _shared_step(self, batch: Dict[str, Any], stage: str, task_id=False):
        batch.setdefault("task_id", "contrastive_classification")

        file_ids_tensor = batch["file_id"]
        if isinstance(file_ids_tensor, torch.Tensor):
            file_ids = file_ids_tensor.view(-1).tolist()
        else:
            file_ids = list(file_ids_tensor)

        first_file_id = file_ids[0]
        meta_row = self.metadata[first_file_id]
        data_name = meta_row.get("Name", "unknown")

        outputs = self.forward(batch)
        logits = outputs["logits"]
        projections = outputs["projection"]

        y = batch["y"]
        ce_loss = self._compute_loss(logits, y)
        y_argmax = torch.argmax(logits, dim=1) if logits.ndim > 1 else logits

        step_metrics: Dict[str, torch.Tensor] = {
            f"{stage}_loss": ce_loss,
            f"{stage}_{data_name}_loss": ce_loss,
        }
        step_metrics.update(self._compute_metrics(y_argmax, y, data_name, stage))

        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != "total":
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        total_loss = ce_loss + reg_dict.get(
            "total", torch.tensor(0.0, device=ce_loss.device)
        )

        contrastive_loss = self._compute_contrastive_loss(
            projections=projections,
            labels=y,
            file_ids=file_ids,
        )

        if contrastive_loss is not None:
            step_metrics[f"{stage}_contrastive_loss"] = contrastive_loss
            weighted = contrastive_loss * self.contrastive_weight
            step_metrics[f"{stage}_contrastive_weighted_loss"] = weighted
            total_loss = total_loss + weighted
            if self.log_components:
                device = logits.device
                step_metrics[f"{stage}_contrastive_temperature"] = torch.as_tensor(
                    self.temperature, device=device
                )
                step_metrics[f"{stage}_contrastive_weight"] = torch.as_tensor(
                    self.contrastive_weight, device=device
                )

        step_metrics[f"{stage}_total_loss"] = total_loss
        step_metrics[f"{stage}_batch_size"] = torch.tensor(y.size(0), device=total_loss.device)

        if self._explainability_enabled and stage in ("val", "test"):
            self._cache_for_explainability(stage, projections.detach(), file_ids, y)

        return step_metrics

    # ------------------------------------------------------------------
    # Contrastive utilities
    # ------------------------------------------------------------------
    def _compute_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor,
        file_ids: list,
    ) -> Optional[torch.Tensor]:
        if self.contrastive_weight <= 0:
            return None
        if projections.size(0) < 2:
            return torch.tensor(0.0, device=projections.device)

        domains = self._resolve_domains_tensor(file_ids, device=projections.device)

        feats = projections
        if self.normalize_embeddings:
            feats = F.normalize(feats, dim=-1)

        # Similarity matrix scaled by temperature
        logits = torch.matmul(feats, feats.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values  # numerical stability
        diag_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, float("-inf"))

        labels = labels.view(-1)
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        if self.mode == "domain-aware":
            domain_mask = domains.unsqueeze(0) != domains.unsqueeze(1)
            positive_mask = label_mask & domain_mask
        else:
            positive_mask = label_mask

        positive_mask = positive_mask & ~diag_mask
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=projections.device)

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (~diag_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1) / torch.clamp(
            positive_counts.float(), min=1.0
        )

        loss = -mean_log_prob_pos[valid].mean()
        return loss

    def _resolve_domains_tensor(self, file_ids: list, device: torch.device) -> torch.Tensor:
        domains = []
        for fid in file_ids:
            meta = self.metadata[fid]
            domains.append(int(meta.get("Domain_id", -1)))
        return torch.as_tensor(domains, device=device)

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._export_stage_explainability('val')

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self._export_stage_explainability('test')

    def _export_stage_explainability(self, stage: str) -> None:
        if not self._explainability_enabled:
            return
        cache = self._cached_embeddings.get(stage, [])
        if not cache:
            return
        base_dir = getattr(self.args_environment, 'output_dir', '.')
        run_name = getattr(self.args_trainer, 'logger_name', None)
        if run_name:
            export_root = str(Path(base_dir) / run_name)
        else:
            export_root = str(Path(base_dir))
        export_tspn_explainability(
            stage=stage,
            cached_batches=cache,
            metadata=self.metadata,
            config=self._explainability_cfg,
            output_dir=export_root,
        )
        self._cached_embeddings[stage].clear()

    # ------------------------------------------------------------------
    # Explainability cache (populated in Task 5)
    # ------------------------------------------------------------------
    def _cache_for_explainability(
        self,
        stage: str,
        projections: torch.Tensor,
        file_ids: list,
        labels: torch.Tensor,
    ) -> None:
        if not self._explainability_enabled:
            return
        self._cached_embeddings.setdefault(stage, []).append(
            {
                "embeddings": projections.cpu(),
                "file_ids": list(file_ids),
                "labels": labels.detach().cpu(),
            }
        )
