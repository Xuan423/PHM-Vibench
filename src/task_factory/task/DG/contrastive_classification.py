"""Domain generalisation classification task with physics-conditioned contrastive learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn

from ...Default_task import Default_task
from ....data_factory.batch import EpisodeBatch
from ...losses import (
    PCCAugmentationConfig,
    PCCBatchBuilder,
    PCCBuilderConfig,
    PCCLossConfig,
    PhysicsConditionedContrastiveLoss,
)





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

        self.lambda_pcc = float(_get(task_cfg, "loss_weight", _get(model_cfg, "loss_weight", 0.0)))
        self.temperature = float(
            max(_get(task_cfg, "temperature", _get(model_cfg, "temperature", 1.0)), 1e-6)
        )
        self.log_components = bool(_get(task_cfg, "log_components", False))
        self.lambda_reg = float(_get(task_cfg, "reg_weight", _get(model_cfg, "reg_weight", 0.0)))

        pcc_cfg = getattr(task_cfg, "pcc", None)
        aug_cfg = getattr(pcc_cfg, "augmentation", None)
        augmentation = PCCAugmentationConfig(
            enable_phi=bool(_get(aug_cfg, "enable_phi", True)),
            time_shift_pct=float(_get(aug_cfg, "time_shift_pct", 0.05)),
            amplitude_range=(
                float(_get(aug_cfg, "amplitude_min", 0.8)),
                float(_get(aug_cfg, "amplitude_max", 1.2)),
            ),
            stretch_range=(
                float(_get(aug_cfg, "stretch_min", 0.95)),
                float(_get(aug_cfg, "stretch_max", 1.05)),
            ),
            ripple_db=float(_get(aug_cfg, "ripple_db", 2.0)),
            views_per_sample=int(_get(aug_cfg, "views_per_sample", 2)),
        )
        builder_config = PCCBuilderConfig(
            lambda_pos=float(_get(pcc_cfg, "lambda_pos", 1.0)),
            lambda_neg=float(_get(pcc_cfg, "lambda_neg", 1.0)),
            cos_threshold_deg=float(_get(pcc_cfg, "cos_threshold_deg", 25.0)),
            augmentation=augmentation,
        )

        self.pcc_builder = PCCBatchBuilder(builder_config, self._encode_for_pcc)
        self.pcc_loss = PhysicsConditionedContrastiveLoss(PCCLossConfig(temperature=self.temperature))

        anchor_cfg = getattr(task_cfg, "weight_anchor", None)
        self.weight_anchor = {
            "enabled": bool(getattr(anchor_cfg, "enabled", False)) if anchor_cfg else False,
            "lambda": float(getattr(anchor_cfg, "lambda", 0.0)) if anchor_cfg else 0.0,
            "modules": list(getattr(anchor_cfg, "modules", [])) if anchor_cfg else [],
        }
        self._anchor_refs: List[Tuple[nn.Parameter, torch.Tensor]] = []
        if self.weight_anchor["enabled"] and self.weight_anchor["lambda"] > 0.0:
            self._init_weight_anchor()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        file_id = batch.get("file_id")
        task_id = batch.get("task_id")
        return self.network(x, file_id, task_id, return_embeddings=True)

    def _encode_for_pcc(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.network.encode(tensor)

    def _shared_step(self, batch: Any, stage: str, task_id=False):
        if isinstance(batch, EpisodeBatch):
            return self._shared_step_episode(batch, stage)
        return self._shared_step_standard(batch, stage)

    def _shared_step_standard(self, batch: Dict[str, Any], stage: str) -> Dict[str, torch.Tensor]:
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
        indicator_penalties = outputs.get("indicator_penalties", {})
        indicator_weights_fn = getattr(self.network, "explain_indicator_weights", None)
        indicator_weights = indicator_weights_fn() if callable(indicator_weights_fn) else None
        projection_matrix_fn = getattr(self.network, "explain_projection_matrix", None)
        projection_matrix = projection_matrix_fn() if callable(projection_matrix_fn) else None

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

        total_loss = ce_loss + reg_dict.get("total", torch.tensor(0.0, device=ce_loss.device))

        if indicator_penalties:
            for name, value in indicator_penalties.items():
                step_metrics[f"{stage}_{name}"] = value

        if indicator_weights and self.log_components:
            weight_matrix = indicator_weights.get("weight")
            if weight_matrix is not None:
                norm_value = torch.as_tensor(weight_matrix.norm(), device=ce_loss.device)
                step_metrics[f"{stage}_indicator_weight_norm"] = norm_value

        metric_reg_value = self._log_metric_regularizer(step_metrics, stage, logits.device)
        if self.lambda_reg > 0:
            metric_reg_loss = metric_reg_value * self.lambda_reg
            step_metrics[f"{stage}_metric_reg_weighted"] = metric_reg_loss
            total_loss = total_loss + metric_reg_loss

        anchor_loss = self._weight_anchor_loss(logits.device)
        if anchor_loss is not None:
            total_loss = total_loss + anchor_loss
            step_metrics[f"{stage}_anchor_loss"] = anchor_loss.detach()

        zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        step_metrics[f"{stage}_pcc_loss"] = zero
        step_metrics[f"{stage}_pcc_weighted_loss"] = zero
        step_metrics[f"{stage}_pcc_positive_count"] = zero
        step_metrics[f"{stage}_pcc_anchor_count"] = zero

        step_metrics[f"{stage}_total_loss"] = total_loss
        step_metrics[f"{stage}_batch_size"] = torch.tensor(y.size(0), device=total_loss.device)

        return step_metrics

    def _shared_step_episode(self, batch: EpisodeBatch, stage: str) -> Dict[str, torch.Tensor]:
        if batch.query_x.size(0) == 0:
            warnings.warn(
                "Episode batch contains no query samples; falling back to flat batch pipeline.",
                RuntimeWarning,
            )
            return self._shared_step_standard(batch.flat_batch, stage)

        device = self.device
        query_x = batch.query_x.to(device)
        query_y = batch.query_y.to(device)
        query_file_ids = self._normalise_file_ids(batch.query_file_ids)
        first_file_id = query_file_ids[0]
        meta_row = self.metadata[first_file_id]
        data_name = meta_row.get("Name", "unknown")

        support_count = batch.support_x.size(0)
        support_embeddings = None
        support_y = batch.support_y.to(device) if support_count > 0 else None
        support_file_ids: List[Any] = []

        if support_count > 0:
            support_file_ids = self._normalise_file_ids(batch.support_file_ids)
            support_outputs = self.network(
                batch.support_x.to(device),
                data_id=support_file_ids,
                task_id="contrastive_classification",
                return_embeddings=True,
            )
            support_embeddings = support_outputs["embeddings"]

        query_outputs = self.network(
            query_x,
            data_id=query_file_ids,
            task_id="contrastive_classification",
            return_embeddings=True,
        )
        logits = query_outputs["logits"]
        query_embeddings = query_outputs["embeddings"]
        indicator_penalties = query_outputs.get("indicator_penalties", {})
        indicator_weights_fn = getattr(self.network, "explain_indicator_weights", None)
        indicator_weights = indicator_weights_fn() if callable(indicator_weights_fn) else None
        projection_matrix_fn = getattr(self.network, "explain_projection_matrix", None)
        projection_matrix = projection_matrix_fn() if callable(projection_matrix_fn) else None

        ce_loss = self._compute_loss(logits, query_y)
        y_argmax = torch.argmax(logits, dim=1) if logits.ndim > 1 else logits

        step_metrics: Dict[str, torch.Tensor] = {
            f"{stage}_loss": ce_loss,
            f"{stage}_{data_name}_loss": ce_loss,
        }
        step_metrics.update(self._compute_metrics(y_argmax, query_y, data_name, stage))

        reg_dict = self._compute_regularization()
        for reg_type, reg_loss_val in reg_dict.items():
            if reg_type != "total":
                step_metrics[f"{stage}_{reg_type}_reg_loss"] = reg_loss_val

        total_loss = ce_loss + reg_dict.get("total", torch.tensor(0.0, device=ce_loss.device))

        if indicator_penalties:
            for name, value in indicator_penalties.items():
                step_metrics[f"{stage}_{name}"] = value

        if indicator_weights and self.log_components:
            weight_matrix = indicator_weights.get("weight")
            if weight_matrix is not None:
                norm_value = torch.as_tensor(weight_matrix.norm(), device=ce_loss.device)
                step_metrics[f"{stage}_indicator_weight_norm"] = norm_value

        metric_reg_value = self._log_metric_regularizer(step_metrics, stage, logits.device)
        if self.lambda_reg > 0:
            metric_reg_loss = metric_reg_value * self.lambda_reg
            step_metrics[f"{stage}_metric_reg_weighted"] = metric_reg_loss
            total_loss = total_loss + metric_reg_loss

        anchor_loss = self._weight_anchor_loss(logits.device)
        if anchor_loss is not None:
            total_loss = total_loss + anchor_loss
            step_metrics[f"{stage}_anchor_loss"] = anchor_loss.detach()

        metric = getattr(self.network, "spd_metric", None)
        if metric is None:
            raise RuntimeError("SPD metric is required for PCC loss but not found on network.")

        pcc_context = self.pcc_builder.build(
            query_embeddings=query_embeddings,
            query_labels=query_y,
            support_embeddings=support_embeddings,
            support_labels=support_y,
            query_x=query_x,
            metric=metric,
        )
        pcc_stats = self.pcc_loss(metric, pcc_context)
        step_metrics[f"{stage}_pcc_loss"] = pcc_stats["loss"]
        step_metrics[f"{stage}_pcc_positive_count"] = pcc_stats["positive_count"]
        step_metrics[f"{stage}_pcc_anchor_count"] = pcc_stats["anchor_count"]

        weighted_pcc = pcc_stats["loss"] * self.lambda_pcc
        step_metrics[f"{stage}_pcc_weighted_loss"] = weighted_pcc
        total_loss = total_loss + weighted_pcc

        step_metrics[f"{stage}_total_loss"] = total_loss
        step_metrics[f"{stage}_batch_size"] = torch.tensor(query_y.size(0), device=total_loss.device)

        return step_metrics

    def _log_metric_regularizer(
        self,
        step_metrics: Dict[str, torch.Tensor],
        stage: str,
        device: torch.device,
    ) -> torch.Tensor:
        metric_fn = getattr(self.network, "metric_regularization", None)
        if not callable(metric_fn):
            return torch.tensor(0.0, device=device)
        value = metric_fn()
        step_metrics[f"{stage}_metric_reg"] = value
        return value

    @staticmethod
    def _normalise_file_ids(file_ids: Sequence[Any]) -> list:
        normalised = []
        for fid in file_ids:
            if isinstance(fid, torch.Tensor):
                normalised.append(fid.item())
            else:
                normalised.append(fid)
        return normalised

    def _init_weight_anchor(self) -> None:
        modules = self.weight_anchor.get("modules") or ["physics_projector", "spd_metric"]
        anchor_pairs: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name in modules:
            module = getattr(self.network, name, None)
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                anchor_pairs.append((param, param.detach().clone()))
        self._anchor_refs = anchor_pairs

    def _weight_anchor_loss(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self._anchor_refs or not self.weight_anchor.get("enabled"):
            return None
        lam = self.weight_anchor.get("lambda", 0.0)
        if lam <= 0:
            return None
        loss = torch.zeros(1, device=device)
        for param, ref in self._anchor_refs:
            ref_tensor = ref
            if ref_tensor.device != param.device:
                ref_tensor = ref_tensor.to(param.device)
            loss = loss + torch.sum((param - ref_tensor) ** 2)
        return loss * lam
