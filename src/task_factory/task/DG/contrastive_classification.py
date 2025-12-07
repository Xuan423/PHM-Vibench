"""Global mini-batch training task for the revamped TSPN contrastive pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ...Default_task import Default_task
from ...losses.tspn_proto import compute_losses
from ... import register_task


@register_task("DG", "contrastive_classification")
class ContrastiveTask(Default_task):
    """Single-phase contrastive classification task with prototype memory."""

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
        self.prototype_memory = getattr(network, "prototype_memory", None)
        if self.prototype_memory is None:
            raise ValueError("TSPNContrastive must expose prototype_memory.")
        self.loss_cfg = network.cfg.loss

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        return self.network(x, return_dict=True)

    def _shared_step(self, batch: Dict[str, Any], stage: str, task_id=False) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        labels = batch["y"].long()
        domains = batch.get("domain", batch.get("d"))
        if domains is not None:
            domains = domains.long()
        if domains is None:
            domains = torch.zeros_like(labels)

        losses = compute_losses(
            logits=outputs["logits"],
            labels=labels,
            u=outputs["u"],
            domains=domains,
            prototype_memory=self.prototype_memory,
            projector_module=self.network.projector,
            loss_cfg=self.loss_cfg,
        )

        metrics = {
            f"{stage}_ce_loss": losses["ce"],
            f"{stage}_proto_loss": losses["proto"],
            f"{stage}_var_loss": losses["var"],
            f"{stage}_ortho_loss": losses["ortho"],
            f"{stage}_phys_loss": losses["phys"],
            f"{stage}_proto_valid": losses["proto_valid"],
            f"{stage}_total_loss": losses["total"],
        }
        return metrics

    def training_step(self, batch: Dict[str, Any], *args, **kwargs) -> STEP_OUTPUT:
        metrics = self._shared_step(batch, "train")
        self._log_metrics(metrics, "train")
        return metrics["train_total_loss"]

    def validation_step(self, batch: Dict[str, Any], *args, **kwargs) -> None:
        metrics = self._shared_step(batch, "val")
        self._log_metrics(metrics, "val")

    def test_step(self, batch: Dict[str, Any], *args, **kwargs) -> None:
        metrics = self._shared_step(batch, "test")
        self._log_metrics(metrics, "test")

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if hasattr(self.network.projector, "maybe_reorthogonalize"):
            self.network.projector.maybe_reorthogonalize(self.global_step)


# Backward compatibility for factory fallback (task_module.task)
task = ContrastiveTask
