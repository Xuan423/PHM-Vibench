from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .feature_schema import FeatureMeta


class DiagnosticsState:
    def __init__(self, feature_meta: List[FeatureMeta], top_t: int = 8, max_members: int = 32) -> None:
        self.feature_meta = feature_meta
        self.top_t = int(top_t)
        self.max_members = int(max_members)
        self._stage_store: Dict[str, Dict[Tuple[str, int, int], dict]] = {}

    def reset_stage(self, stage: str) -> None:
        self._stage_store[stage] = {}

    def update(self, stage: str, extras: Dict[str, torch.Tensor | str | None]) -> None:
        if stage not in self._stage_store:
            self.reset_stage(stage)
        assignments = extras.get("prototype_assignments")
        labels = extras.get("labels")
        file_ids = extras.get("file_ids")
        h_raw = extras.get("h_raw")
        head_key = str(extras.get("head_key", "default"))
        if assignments is None or labels is None or file_ids is None or h_raw is None:
            return

        assignments = assignments.detach().cpu()
        labels = labels.detach().cpu()
        file_ids = file_ids.detach().cpu()
        h_raw = h_raw.detach().cpu()
        store = self._stage_store[stage]
        for index in range(assignments.shape[0]):
            key = (head_key, int(labels[index].item()), int(assignments[index].item()))
            item = store.setdefault(key, {"file_ids": [], "vectors": [], "count": 0})
            item["count"] += 1
            if len(item["file_ids"]) < self.max_members:
                item["file_ids"].append(int(file_ids[index].item()))
                item["vectors"].append(h_raw[index])

    def _feature_stats(self, vectors: List[torch.Tensor], feature_indices: List[int]) -> List[dict]:
        if not vectors:
            return []
        stacked = torch.stack(vectors, dim=0)
        stats = []
        for feature_index in feature_indices:
            values = stacked[:, feature_index]
            stats.append(
                {
                    "feature_index": int(feature_index),
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                    "q25": float(torch.quantile(values, 0.25).item()),
                    "q50": float(torch.quantile(values, 0.50).item()),
                    "q75": float(torch.quantile(values, 0.75).item()),
                }
            )
        return stats

    def build_feature_map(self) -> List[dict]:
        return [item.to_dict() for item in self.feature_meta]

    def build_stage_payload(self, stage: str, prototype_bank, metric_weight: torch.Tensor) -> Dict[str, object]:
        store = self._stage_store.get(stage, {})
        feature_map = self.build_feature_map()
        health = prototype_bank.export_all_health()
        prototype_cards: List[dict] = []
        for head_key in prototype_bank.head_keys:
            signatures = prototype_bank.compute_signatures(head_key, metric_weight).detach().cpu()
            head_health = health[head_key]
            num_classes = signatures.shape[0]
            num_protos = signatures.shape[1]
            for class_id in range(num_classes):
                class_signatures = signatures[class_id]
                for proto_id in range(num_protos):
                    signature = class_signatures[proto_id]
                    top_indices = torch.topk(
                        signature,
                        k=min(self.top_t, signature.numel()),
                    ).indices.tolist()
                    signature_sum = signature.sum().item()
                    if signature_sum > 0:
                        probs = signature / signature_sum
                        entropy = float(-(probs * probs.clamp_min(1e-6).log()).sum().item())
                        effective_feature_count = float(torch.exp(torch.as_tensor(entropy)).item())
                    else:
                        entropy = 0.0
                        effective_feature_count = 0.0
                    pairwise = torch.nn.functional.normalize(class_signatures, dim=-1)
                    cosine = torch.matmul(pairwise, pairwise.transpose(0, 1))
                    overlap = 0.0
                    if num_protos > 1:
                        max_same = float(
                            cosine[proto_id, torch.arange(num_protos) != proto_id].max().item()
                        )
                    else:
                        max_same = 0.0
                    item = store.get((head_key, class_id, proto_id), {"file_ids": [], "vectors": [], "count": 0})
                    prototype_cards.append(
                        {
                            "stage": stage,
                            "head_key": head_key,
                            "class_id": class_id,
                            "prototype_id": proto_id,
                            "member_file_ids": item["file_ids"],
                            "top_features": [
                                {
                                    "feature_index": int(feature_index),
                                    "display_name": self.feature_meta[int(feature_index)].display_name,
                                    "weight": float(signature[int(feature_index)].item()),
                                }
                                for feature_index in top_indices
                            ],
                            "feature_stats": self._feature_stats(item["vectors"], [int(idx) for idx in top_indices]),
                            "readability_metrics": {
                                "effective_feature_count": effective_feature_count,
                                "normalized_entropy": entropy,
                            },
                            "complementarity_metrics": {
                                "max_same_class_cos": max_same,
                                "top_feature_overlap": overlap,
                            },
                            "occupancy_metrics": {
                                "assignment_count": int(item["count"]),
                                "assignment_ratio": float(head_health["max_assignment_ratio"][class_id]),
                            },
                        }
                    )
        return {
            "feature_map": feature_map,
            "prototype_health": health,
            "prototype_cards": prototype_cards,
        }
