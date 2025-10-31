"""Domain generalisation classification task with contrastive learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F

from ...Default_task import Default_task
from ....data_factory.batch import EpisodeBatch, EpisodeLabelView
from ....utils.explainability import export_tspn_explainability


@dataclass
class ContrastiveBranchStat:
    name: str
    loss: torch.Tensor
    positive_count: torch.Tensor
    anchor_count: torch.Tensor
    weight: Optional[torch.Tensor] = None


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

        model_support_cfg = getattr(model_cfg, "support_loss", None)
        support_cfg = getattr(task_cfg, "support_loss", None)
        self.support_loss_mode = str(
            _get(support_cfg, "mode", _get(model_support_cfg, "mode", "none"))
        ).lower()
        self.support_loss_weight = float(
            _get(support_cfg, "loss_weight", _get(model_support_cfg, "loss_weight", 0.0))
        )
        self.support_detach_prototypes = bool(
            _get(
                support_cfg,
                "detach_prototypes",
                _get(model_support_cfg, "detach_prototypes", False),
            )
        )
        self.support_prototype_metric = str(
            _get(
                support_cfg,
                "prototype_metric",
                _get(model_support_cfg, "prototype_metric", "cosine"),
            )
        ).lower()
        self.support_log_prefix = str(
            _get(support_cfg, "log_prefix", _get(model_support_cfg, "log_prefix", "support"))
        )
        self._support_component_weights = {
            "support_ce": float(
                _get(support_cfg, "ce_weight", _get(model_support_cfg, "ce_weight", 1.0))
            ),
            "support_proto": float(
                _get(
                    support_cfg,
                    "prototype_weight",
                    _get(model_support_cfg, "prototype_weight", 1.0),
                )
            ),
        }

        model_weighting_cfg = getattr(model_cfg, "weighting", None)
        weighting_cfg = getattr(task_cfg, "weighting", None)
        self.weighting_strategy = str(
            _get(weighting_cfg, "strategy", _get(model_weighting_cfg, "strategy", "uniform"))
        ).lower()
        self.weighting_alpha = float(
            _get(weighting_cfg, "alpha", _get(model_weighting_cfg, "alpha", 1.0))
        )
        self.weighting_ema = float(
            _get(weighting_cfg, "ema", _get(model_weighting_cfg, "ema", 0.9))
        )
        self.weighting_temperature = float(
            max(
                _get(
                    weighting_cfg,
                    "temperature",
                    _get(model_weighting_cfg, "temperature", 1.0),
                ),
                1e-6,
            )
        )
        self._branch_loss_ema: Dict[str, float] = {}
        self._branch_sq_ema: Dict[str, float] = {}

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

        total_loss = ce_loss + reg_dict.get(
            "total", torch.tensor(0.0, device=ce_loss.device)
        )

        if indicator_penalties:
            penalty_total = torch.tensor(0.0, device=ce_loss.device)
            for name, value in indicator_penalties.items():
                penalty_total = penalty_total + value
                step_metrics[f"{stage}_{name}"] = value
            step_metrics[f"{stage}_indicator_penalty_total"] = penalty_total
            total_loss = total_loss + penalty_total

        if indicator_weights and self.log_components:
            weight_matrix = indicator_weights.get("weight")
            if weight_matrix is not None:
                norm_value = torch.as_tensor(weight_matrix.norm(), device=ce_loss.device)
                step_metrics[f"{stage}_indicator_weight_norm"] = norm_value

        query_domains = self._resolve_domains_tensor(file_ids, device=logits.device)
        branch_stats = self._build_contrastive_branches(
            support_projection=None,
            support_labels=None,
            support_domains=None,
            query_projection=projections,
            query_labels=y,
            query_domains=query_domains,
        )
        contrastive_loss, branch_weights = self._combine_contrastive_branches(branch_stats)

        for name, stat in branch_stats.items():
            step_metrics[f"{stage}_contrastive_{name}_loss"] = stat.loss
            step_metrics[f"{stage}_contrastive_{name}_positives"] = stat.positive_count
            step_metrics[f"{stage}_contrastive_{name}_anchors"] = stat.anchor_count
            weight = branch_weights.get(name)
            if weight is not None:
                step_metrics[f"{stage}_contrastive_{name}_weight"] = weight

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

        indicator_snapshot = None
        if indicator_weights or branch_weights or projection_matrix is not None:
            indicator_snapshot = {}
            if indicator_weights:
                indicator_snapshot["indicator_weights"] = {
                    key: tensor.cpu() for key, tensor in indicator_weights.items()
                }
            if projection_matrix is not None:
                indicator_snapshot["projection_matrix"] = projection_matrix.cpu()
            if branch_weights:
                indicator_snapshot["branch_weights"] = {
                    name: weight.detach().cpu() for name, weight in branch_weights.items()
                }

        if self._explainability_enabled and stage in ("val", "test"):
            self._cache_for_explainability(
                stage,
                projections.detach(),
                file_ids,
                y,
                indicator_snapshot=indicator_snapshot,
            )

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
        support_file_ids: List[Any] = []
        support_y = batch.support_y.to(device) if support_count > 0 else None
        support_projection = None
        support_embeddings = None
        support_logits = None
        support_losses: Dict[str, torch.Tensor] = {}

        if support_count > 0:
            support_file_ids = self._normalise_file_ids(batch.support_file_ids)
            support_outputs = self.network(
                batch.support_x.to(device),
                data_id=support_file_ids,
                task_id="contrastive_classification",
                return_embeddings=True,
            )
            support_logits = support_outputs["logits"]
            support_embeddings = support_outputs["embeddings"]
            support_projection = support_outputs["projection"]

        prototype_tensor = self._compute_prototypes(support_embeddings, batch.label_views)

        query_outputs = self.network(
            query_x,
            data_id=query_file_ids,
            task_id="contrastive_classification",
            return_embeddings=True,
        )
        logits = query_outputs["logits"]
        query_embeddings = query_outputs["embeddings"]
        query_projection = query_outputs["projection"]
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

        if (
            support_count > 0
            and self.support_loss_weight > 0
            and self.support_loss_mode in {"cross_entropy", "prototype", "hybrid"}
        ):
            support_losses = self._compute_support_losses(
                logits=support_logits,
                embeddings=support_embeddings,
                labels=support_y,
                label_views=batch.label_views,
            )

        if support_losses:
            combined_support = torch.tensor(0.0, device=ce_loss.device)
            for key, loss_val in support_losses.items():
                component_weight = self._support_component_weights.get(key, 1.0)
                step_metrics[f"{stage}_{key}"] = loss_val
                weighted_component = loss_val * component_weight
                step_metrics[f"{stage}_{key}_weighted"] = weighted_component
                combined_support = combined_support + weighted_component
            weighted_support_loss = combined_support * self.support_loss_weight
            step_metrics[f"{stage}_{self.support_log_prefix}_loss"] = combined_support
            step_metrics[f"{stage}_{self.support_log_prefix}_weighted_loss"] = weighted_support_loss
            total_loss = total_loss + weighted_support_loss

        if indicator_penalties:
            penalty_total = torch.tensor(0.0, device=ce_loss.device)
            for name, value in indicator_penalties.items():
                penalty_total = penalty_total + value
                step_metrics[f"{stage}_{name}"] = value
            step_metrics[f"{stage}_indicator_penalty_total"] = penalty_total
            total_loss = total_loss + penalty_total

        if indicator_weights and self.log_components:
            weight_matrix = indicator_weights.get("weight")
            if weight_matrix is not None:
                norm_value = torch.as_tensor(weight_matrix.norm(), device=ce_loss.device)
                step_metrics[f"{stage}_indicator_weight_norm"] = norm_value

        support_domains = (
            self._resolve_domains_tensor(support_file_ids, device=ce_loss.device)
            if support_count > 0
            else None
        )
        query_domains = self._resolve_domains_tensor(query_file_ids, device=ce_loss.device)
        branch_stats = self._build_contrastive_branches(
            support_projection=support_projection,
            support_labels=support_y,
            support_domains=support_domains,
            query_projection=query_projection,
            query_labels=query_y,
            query_domains=query_domains,
        )
        contrastive_loss, branch_weights = self._combine_contrastive_branches(branch_stats)

        for name, stat in branch_stats.items():
            step_metrics[f"{stage}_contrastive_{name}_loss"] = stat.loss
            step_metrics[f"{stage}_contrastive_{name}_positives"] = stat.positive_count
            step_metrics[f"{stage}_contrastive_{name}_anchors"] = stat.anchor_count
            weight = branch_weights.get(name)
            if weight is not None:
                step_metrics[f"{stage}_contrastive_{name}_weight"] = weight

        if contrastive_loss is not None:
            step_metrics[f"{stage}_contrastive_loss"] = contrastive_loss
            weighted = contrastive_loss * self.contrastive_weight
            step_metrics[f"{stage}_contrastive_weighted_loss"] = weighted
            total_loss = total_loss + weighted
            if self.log_components:
                device_for_logs = logits.device
                step_metrics[f"{stage}_contrastive_temperature"] = torch.as_tensor(
                    self.temperature, device=device_for_logs
                )
                step_metrics[f"{stage}_contrastive_weight"] = torch.as_tensor(
                    self.contrastive_weight, device=device_for_logs
                )

        if prototype_tensor is not None and self.log_components:
            step_metrics[f"{stage}_prototype_norm_mean"] = prototype_tensor.norm(dim=-1).mean()

        step_metrics[f"{stage}_total_loss"] = total_loss
        step_metrics[f"{stage}_batch_size"] = torch.tensor(query_y.size(0), device=total_loss.device)

        indicator_snapshot = None
        if indicator_weights or branch_weights or projection_matrix is not None:
            indicator_snapshot = {}
            if indicator_weights:
                indicator_snapshot["indicator_weights"] = {
                    key: tensor.cpu() for key, tensor in indicator_weights.items()
                }
            if projection_matrix is not None:
                indicator_snapshot["projection_matrix"] = projection_matrix.cpu()
            if branch_weights:
                indicator_snapshot["branch_weights"] = {
                    name: weight.detach().cpu() for name, weight in branch_weights.items()
                }

        if self._explainability_enabled and stage in ("val", "test"):
            self._cache_for_explainability(
                stage,
                query_projection.detach(),
                query_file_ids,
                query_y,
                indicator_snapshot=indicator_snapshot,
            )

        return step_metrics

    @staticmethod
    def _normalise_file_ids(file_ids: Sequence[Any]) -> list:
        normalised = []
        for fid in file_ids:
            if isinstance(fid, torch.Tensor):
                normalised.append(fid.item())
            else:
                normalised.append(fid)
        return normalised

    @staticmethod
    def _compute_prototypes(
        support_embeddings: Optional[torch.Tensor],
        label_views: Sequence[EpisodeLabelView],
    ) -> Optional[torch.Tensor]:
        if support_embeddings is None or support_embeddings.size(0) == 0:
            return None
        prototypes = []
        for view in label_views:
            if view.support_count == 0:
                continue
            proto = support_embeddings[view.support_slice].mean(dim=0)
            prototypes.append(proto)
        if not prototypes:
            return None
        return torch.stack(prototypes, dim=0)

    # ------------------------------------------------------------------
    # Contrastive utilities
    # ------------------------------------------------------------------
    def _compute_support_losses(
        self,
        *,
        logits: Optional[torch.Tensor],
        embeddings: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        label_views: Sequence[EpisodeLabelView],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        if labels is None or labels.numel() == 0:
            return losses

        mode = self.support_loss_mode
        if mode in {"cross_entropy", "hybrid"} and logits is not None:
            losses["support_ce"] = self._compute_loss(logits, labels)

        if mode in {"prototype", "hybrid"} and embeddings is not None:
            prototypes = self._compute_prototypes(embeddings, label_views)
            if prototypes is not None:
                if self.support_detach_prototypes:
                    prototypes = prototypes.detach()
                proto_loss = self._prototype_alignment_loss(embeddings, prototypes, label_views)
                losses["support_proto"] = proto_loss

        return losses

    def _prototype_alignment_loss(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        label_views: Sequence[EpisodeLabelView],
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        proto_idx = 0
        for view in label_views:
            if view.support_count == 0:
                continue
            samples = embeddings[view.support_slice]
            if samples.numel() == 0:
                proto_idx += 1
                continue
            prototype_vec = prototypes[proto_idx]
            proto_idx += 1
            if self.support_prototype_metric == "cosine":
                similarity = F.cosine_similarity(samples, prototype_vec.unsqueeze(0), dim=-1)
                losses.append(1.0 - similarity.mean())
            else:
                losses.append(
                    F.mse_loss(
                        samples,
                        prototype_vec.unsqueeze(0).expand_as(samples),
                    )
                )
        if not losses:
            return torch.tensor(0.0, device=embeddings.device)
        return torch.stack(losses).mean()

    def _build_contrastive_branches(
        self,
        *,
        support_projection: Optional[torch.Tensor],
        support_labels: Optional[torch.Tensor],
        support_domains: Optional[torch.Tensor],
        query_projection: Optional[torch.Tensor],
        query_labels: Optional[torch.Tensor],
        query_domains: Optional[torch.Tensor],
    ) -> Dict[str, ContrastiveBranchStat]:
        branches: Dict[str, ContrastiveBranchStat] = {}

        query_stat = self._contrastive_branch(
            name="query_query",
            anchors=query_projection,
            anchor_labels=query_labels,
            anchor_domains=query_domains,
            candidates=query_projection,
            candidate_labels=query_labels,
            candidate_domains=query_domains,
            exclude_self=True,
        )
        if query_stat is not None:
            branches[query_stat.name] = query_stat

        if support_projection is not None and support_labels is not None and support_labels.numel() > 0:
            support_stat = self._contrastive_branch(
                name="support_support",
                anchors=support_projection,
                anchor_labels=support_labels,
                anchor_domains=support_domains,
                candidates=support_projection,
                candidate_labels=support_labels,
                candidate_domains=support_domains,
                exclude_self=True,
            )
            if support_stat is not None:
                branches[support_stat.name] = support_stat

            cross_stat = self._contrastive_cross_branch(
                name="support_query",
                support_projection=support_projection,
                support_labels=support_labels,
                support_domains=support_domains,
                query_projection=query_projection,
                query_labels=query_labels,
                query_domains=query_domains,
            )
            if cross_stat is not None:
                branches[cross_stat.name] = cross_stat

        return branches

    def _contrastive_cross_branch(
        self,
        *,
        name: str,
        support_projection: Optional[torch.Tensor],
        support_labels: Optional[torch.Tensor],
        support_domains: Optional[torch.Tensor],
        query_projection: Optional[torch.Tensor],
        query_labels: Optional[torch.Tensor],
        query_domains: Optional[torch.Tensor],
    ) -> Optional[ContrastiveBranchStat]:
        forward = self._contrastive_branch(
            name=f"{name}_forward",
            anchors=support_projection,
            anchor_labels=support_labels,
            anchor_domains=support_domains,
            candidates=query_projection,
            candidate_labels=query_labels,
            candidate_domains=query_domains,
            exclude_self=False,
        )
        backward = self._contrastive_branch(
            name=f"{name}_backward",
            anchors=query_projection,
            anchor_labels=query_labels,
            anchor_domains=query_domains,
            candidates=support_projection,
            candidate_labels=support_labels,
            candidate_domains=support_domains,
            exclude_self=False,
        )

        if forward is None and backward is None:
            return None

        components: List[ContrastiveBranchStat] = [b for b in (forward, backward) if b is not None]
        losses = torch.stack([comp.loss for comp in components])
        combined_loss = losses.mean()
        combined_positive = sum(comp.positive_count for comp in components)
        combined_anchor = sum(comp.anchor_count for comp in components)

        return ContrastiveBranchStat(
            name=name,
            loss=combined_loss,
            positive_count=combined_positive.to(combined_loss),
            anchor_count=combined_anchor.to(combined_loss),
        )

    def _contrastive_branch(
        self,
        *,
        name: str,
        anchors: Optional[torch.Tensor],
        anchor_labels: Optional[torch.Tensor],
        anchor_domains: Optional[torch.Tensor],
        candidates: Optional[torch.Tensor],
        candidate_labels: Optional[torch.Tensor],
        candidate_domains: Optional[torch.Tensor],
        exclude_self: bool,
    ) -> Optional[ContrastiveBranchStat]:
        if (
            anchors is None
            or candidates is None
            or anchor_labels is None
            or candidate_labels is None
            or anchors.size(0) == 0
            or candidates.size(0) == 0
        ):
            return None

        feats_anchor = anchors
        feats_candidate = candidates
        if self.normalize_embeddings:
            feats_anchor = F.normalize(feats_anchor, dim=-1)
            feats_candidate = F.normalize(feats_candidate, dim=-1)

        logits = torch.matmul(feats_anchor, feats_candidate.T) / self.temperature
        logits = logits - torch.max(logits, dim=1, keepdim=True).values

        positive_mask = anchor_labels.view(-1, 1) == candidate_labels.view(1, -1)
        if self.mode == "domain-aware" and anchor_domains is not None and candidate_domains is not None:
            domain_mask = anchor_domains.view(-1, 1) != candidate_domains.view(1, -1)
            positive_mask = positive_mask & domain_mask

        if exclude_self and anchors.size(0) == candidates.size(0):
            diag = torch.eye(anchors.size(0), device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(diag, float("-inf"))
            positive_mask = positive_mask & ~diag

        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not torch.any(valid):
            return None

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
        mean_log_prob_pos = masked_log_prob.sum(dim=1) / positive_counts.clamp(min=1.0).float()
        loss = -mean_log_prob_pos[valid].mean()

        return ContrastiveBranchStat(
            name=name,
            loss=loss,
            positive_count=positive_counts[valid].sum().to(loss),
            anchor_count=torch.as_tensor(float(valid.sum().item()), device=loss.device, dtype=loss.dtype),
        )

    def _combine_contrastive_branches(
        self, branches: Dict[str, ContrastiveBranchStat]
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not branches:
            return None, {}

        device = next(iter(branches.values())).loss.device
        dtype = next(iter(branches.values())).loss.dtype

        total_positive = sum(float(stat.positive_count.detach().item()) for stat in branches.values())
        if total_positive > 0:
            base_weights = {
                name: float(stat.positive_count.detach().item()) / total_positive
                for name, stat in branches.items()
            }
        else:
            uniform = 1.0 / len(branches)
            base_weights = {name: uniform for name in branches}

        if self.weighting_strategy == "gradnorm":
            weight_dict = self._gradnorm_weights(branches, base_weights)
        elif self.weighting_strategy == "uncertainty":
            weight_dict = self._uncertainty_weights(branches, base_weights)
        elif self.weighting_strategy in {"uniform", "none"}:
            weight_dict = base_weights
        else:
            weight_dict = {
                name: weight ** self.weighting_alpha for name, weight in base_weights.items()
            }

        total_weight = sum(weight_dict.values())
        if total_weight <= 0:
            weight_dict = {name: 1.0 / len(branches) for name in branches}
            total_weight = 1.0

        normalised = {name: weight / total_weight for name, weight in weight_dict.items()}
        branch_weights = {
            name: torch.as_tensor(weight, device=device, dtype=dtype) for name, weight in normalised.items()
        }

        if self.contrastive_weight <= 0:
            for name, weight in branch_weights.items():
                branches[name].weight = weight
            return None, branch_weights

        weighted_losses = []
        for name, stat in branches.items():
            weight = branch_weights.get(name, torch.tensor(0.0, device=device, dtype=dtype))
            stat.weight = weight
            weighted_losses.append(stat.loss * weight)

        combined_loss = torch.stack(weighted_losses).sum()
        return combined_loss, branch_weights

    def _gradnorm_weights(
        self,
        branches: Dict[str, ContrastiveBranchStat],
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        eps = 1e-8
        weights: Dict[str, float] = {}
        for name, stat in branches.items():
            loss_val = abs(float(stat.loss.detach().item()))
            ema = self._branch_loss_ema.get(name, loss_val)
            ema = self.weighting_ema * ema + (1.0 - self.weighting_ema) * loss_val
            self._branch_loss_ema[name] = ema
            ratio = (loss_val / (ema + eps)) ** self.weighting_alpha
            weights[name] = base_weights.get(name, 0.0) * ratio
        return weights

    def _uncertainty_weights(
        self,
        branches: Dict[str, ContrastiveBranchStat],
        base_weights: Dict[str, float],
    ) -> Dict[str, float]:
        eps = 1e-8
        weights: Dict[str, float] = {}
        for name, stat in branches.items():
            loss_val = abs(float(stat.loss.detach().item()))
            ema = self._branch_loss_ema.get(name, loss_val)
            ema = self.weighting_ema * ema + (1.0 - self.weighting_ema) * loss_val
            self._branch_loss_ema[name] = ema
            sq_ema = self._branch_sq_ema.get(name, loss_val**2)
            sq_ema = self.weighting_ema * sq_ema + (1.0 - self.weighting_ema) * (loss_val**2)
            self._branch_sq_ema[name] = sq_ema
            variance = max(sq_ema - ema**2, eps)
            weights[name] = base_weights.get(name, 0.0) / variance
        return weights

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
        *,
        indicator_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._explainability_enabled:
            return
        payload: Dict[str, Any] = {
            "embeddings": projections.cpu(),
            "file_ids": list(file_ids),
            "labels": labels.detach().cpu(),
        }
        if indicator_snapshot:
            payload.update(indicator_snapshot)
        self._cached_embeddings.setdefault(stage, []).append(payload)
