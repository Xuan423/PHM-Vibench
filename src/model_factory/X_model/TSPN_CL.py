"""TSPN-CL: TSPN backbone with concept head, stability top-K, and prototype InfoNCE."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TSPN import Model as TSPNBackbone
from .utils.concept_projector import ConceptProjector
from .utils.topk_selector import StabilityTopKSelector


class Model(nn.Module):
    def __init__(self, args: Any, metadata: Any = None) -> None:
        super().__init__()
        self.args = args
        self.metadata = metadata

        # Defaults for new flags
        self.use_concept = getattr(args, "use_physical_concept_head", False) or getattr(
            args, "use_contrastive_head", False
        )
        self.use_contrastive = getattr(args, "use_contrastive_head", False)
        self.top_k = getattr(args, "top_k_features", None)
        self.concept_dim = getattr(args, "concept_dim", 32)
        self.lambda_contrastive = getattr(args, "lambda_contrastive", 0.1)
        self.temperature = getattr(args, "temperature", 0.07)
        self.lambda_phys = getattr(args, "lambda_phys_consistency", 0.0)
        self.topk_score_mode = getattr(args, "topk_score_mode", "fisher_over_domain_var")
        self.topk_ema_momentum = getattr(args, "topk_ema_momentum", 0.9)
        self.topk_warmup_epochs = getattr(args, "topk_warmup_epochs", 1)

        # Backbone (reuses TSPN layers; classifier not used)
        self.backbone = TSPNBackbone(args, metadata)

        feature_dim = getattr(self.backbone, "channel_for_classifier", None)
        if feature_dim is None:
            raise ValueError("Backbone missing channel_for_classifier to define feature dimension.")

        if self.use_concept:
            if self.top_k is None:
                raise ValueError("top_k_features must be set when concept head is enabled.")
            self.topk_selector = StabilityTopKSelector(
                top_k=self.top_k,
                score_mode=self.topk_score_mode,
                ema_momentum=self.topk_ema_momentum,
                warmup_epochs=self.topk_warmup_epochs,
            )
            self.projector = ConceptProjector(self.top_k, self.concept_dim)
            classifier_in = self.concept_dim
        else:
            self.topk_selector = None
            self.projector = None
            classifier_in = feature_dim

        self.heads = self._build_heads(args.num_classes, classifier_in)

    def _build_heads(self, num_classes: Any, in_dim: int) -> nn.Module:
        if isinstance(num_classes, dict):
            module_dict = nn.ModuleDict()
            for k, v in num_classes.items():
                # module_dict[str(k)] = nn.Sequential(
                #     nn.Linear(in_dim, 128),
                #     nn.ReLU(),
                #     nn.Linear(128, int(v)),
                # )
                module_dict[str(k)] = nn.Linear(in_dim, int(v))
            return module_dict
        return nn.ModuleDict({"default": nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, int(num_classes)))})

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone up to feature extractor, returning h_raw."""
        h = x
        for layer in self.backbone.signal_processing_layers:
            h = layer(h)
        h = self.backbone.feature_extractor_layers(h)
        return h

    def _get_domains(self, file_ids: torch.Tensor) -> torch.Tensor:
        if self.metadata is None:
            return torch.zeros_like(file_ids, dtype=torch.long)
        dom_list = []
        for fid in file_ids.view(-1).tolist():
            try:
                dom = self.metadata[fid].get("Domain_id", 0)
            except Exception:
                dom = 0
            dom_list.append(int(dom))
        return torch.tensor(dom_list, device=file_ids.device, dtype=torch.long)

    def _pick_head(self, file_ids: Optional[torch.Tensor]) -> nn.Module:
        if len(self.heads) == 1:
            return next(iter(self.heads.values()))
        dataset_id = None
        if file_ids is not None and self.metadata is not None:
            fid0 = int(file_ids.view(-1)[0].item())
            try:
                dataset_id = str(self.metadata[fid0].get("Dataset_id"))
            except Exception:
                dataset_id = None
        if dataset_id and dataset_id in self.heads:
            return self.heads[dataset_id]
        # fallback to first head deterministically
        return self.heads[sorted(self.heads.keys())[0]]

    def forward(self, x: torch.Tensor, file_id=None, task_id=None):
        """Legacy forward: returns logits only."""
        h_raw = self._extract_features(x)
        head = self._pick_head(None)
        return head(h_raw)

    def forward_with_batch(self, batch: Dict[str, Any], epoch: int = 0) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        labels = batch.get("y")
        file_ids_raw = batch.get("_file_ids_raw", batch.get("file_id"))
        file_ids_tensor = self._to_tensor_ids(file_ids_raw, device=x.device)
        domains = self._get_domains(file_ids_tensor)

        h_raw = self._extract_features(x)

        if not self.use_concept:
            head = self._pick_head(file_ids_tensor)
            logits = head(h_raw)
            return {"logits": logits}

        if labels is None:
            # In inference, skip scoring update if labels missing
            h_sel = h_raw[:, : self.top_k]
            idx = torch.arange(h_sel.shape[1], device=h_sel.device)
        else:
            h_sel, idx = self.topk_selector.select(h_raw, labels.to(h_raw.device), domains, epoch)

        c, W = self.projector(h_sel)
        head = self._pick_head(file_ids_tensor)
        logits = head(c)

        return {
            "logits": logits,
            "c": c,
            "h_sel": h_sel,
            "W": W,
            "domains": domains,
            "labels": labels,
        }

    def _to_tensor_ids(self, file_ids_raw: Any, device: torch.device) -> torch.Tensor:
        if file_ids_raw is None:
            return torch.zeros(1, device=device, dtype=torch.long)
        if torch.is_tensor(file_ids_raw):
            return file_ids_raw.to(device).long()
        try:
            return torch.tensor(file_ids_raw, device=device, dtype=torch.long)
        except Exception:
            return torch.zeros(1, device=device, dtype=torch.long)

    def compute_contrastive_loss(self, extras: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        c = extras["c"]
        h_sel = extras["h_sel"]
        W = extras["W"]
        labels = labels.to(c.device)

        proto_labels, prototypes = self._compute_prototypes(c, labels)
        L_info = self._info_nce(c, prototypes, labels, proto_labels)
        L_phys = self._physical_consistency(h_sel, prototypes, proto_labels, W, labels)
        return L_info + self.lambda_phys * L_phys, L_info, L_phys

    def _compute_prototypes(self, c: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uniq = labels.unique()
        protos = []
        proto_labels = []
        for lbl in uniq:
            mask = labels == lbl
            protos.append(c[mask].mean(dim=0))
            proto_labels.append(lbl)
        return torch.stack(proto_labels).to(c.device), torch.stack(protos)

    def _info_nce(
        self,
        c: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
        proto_labels: torch.Tensor,
    ) -> torch.Tensor:
        # cosine similarity
        sim = F.cosine_similarity(
            c.unsqueeze(1),
            prototypes.unsqueeze(0),
            dim=-1,
        )  # [B, P]
        logits = sim / self.temperature
        # map each label to proto index
        label_to_idx = {int(lbl.item()): idx for idx, lbl in enumerate(proto_labels)}
        target = torch.tensor([label_to_idx[int(l.item())] for l in labels], device=c.device)
        return F.cross_entropy(logits, target)

    def _physical_consistency(
        self,
        h_sel: torch.Tensor,
        prototypes: torch.Tensor,
        proto_labels: torch.Tensor,
        W: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # project prototypes to physical space
        p_phys = torch.matmul(W.t(), prototypes.t()).t()  # [P, K]
        p_phys = F.normalize(p_phys, dim=-1)
        mu_phys = []
        for lbl in proto_labels:
            mask = labels == lbl
            if mask.sum() == 0:
                mu = torch.zeros_like(h_sel[0])
            else:
                mu = h_sel[mask].mean(dim=0)
            mu = F.normalize(mu, dim=-1)
            mu_phys.append(mu)
        mu_phys = torch.stack(mu_phys, dim=0)  # [P, K]
        return torch.norm(p_phys - mu_phys, dim=1).mean()
