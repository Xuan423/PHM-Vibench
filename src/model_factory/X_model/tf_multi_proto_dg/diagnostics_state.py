from __future__ import annotations

from typing import Dict, List

import torch

from .explain_export import (
    build_alpha_report,
    build_compact_concept_report,
    build_patch_band_report,
    build_role_basis_report,
)
from .feature_schema import FeatureMeta, build_basis_labels


class DiagnosticsState:
    def __init__(
        self,
        feature_meta: List[FeatureMeta],
        time_operator_names: List[str],
        freq_operator_names: List[str],
        time_indicator_names: List[str],
        freq_indicator_names: List[str],
        role_dim: int,
        top_t: int = 8,
        max_members: int = 32,
    ) -> None:
        self.feature_meta = feature_meta
        self.time_operator_names = list(time_operator_names)
        self.freq_operator_names = list(freq_operator_names)
        self.time_indicator_names = list(time_indicator_names)
        self.freq_indicator_names = list(freq_indicator_names)
        self.role_dim = int(role_dim)
        self.top_t = int(top_t)
        self.max_members = int(max_members)
        self._stage_records: Dict[str, List[dict]] = {}

    def reset_stage(self, stage: str) -> None:
        self._stage_records[stage] = []

    def update(self, stage: str, extras: Dict[str, torch.Tensor | str | None]) -> None:
        if stage not in self._stage_records:
            self.reset_stage(stage)
        stage_records = self._stage_records[stage]
        if len(stage_records) >= self.max_members:
            return

        labels = extras.get("labels")
        file_ids = extras.get("file_ids")
        target_class_ids = extras.get("target_class_ids")
        record = {
            "file_ids": file_ids.detach().cpu() if torch.is_tensor(file_ids) else None,
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else None,
            "target_class_ids": (
                target_class_ids.detach().cpu() if torch.is_tensor(target_class_ids) else None
            ),
            "alpha_t": extras.get("alpha_t").detach().cpu() if torch.is_tensor(extras.get("alpha_t")) else None,
            "alpha_f": extras.get("alpha_f").detach().cpu() if torch.is_tensor(extras.get("alpha_f")) else None,
            "A_t": extras.get("A_t").detach().cpu() if torch.is_tensor(extras.get("A_t")) else None,
            "A_f": extras.get("A_f").detach().cpu() if torch.is_tensor(extras.get("A_f")) else None,
            "S": extras.get("S").detach().cpu() if torch.is_tensor(extras.get("S")) else None,
            "w_t": extras.get("w_t").detach().cpu() if torch.is_tensor(extras.get("w_t")) else None,
            "w_f": extras.get("w_f").detach().cpu() if torch.is_tensor(extras.get("w_f")) else None,
            "g_t": extras.get("g_t").detach().cpu() if torch.is_tensor(extras.get("g_t")) else None,
            "g_f": extras.get("g_f").detach().cpu() if torch.is_tensor(extras.get("g_f")) else None,
            "h": extras.get("h").detach().cpu() if torch.is_tensor(extras.get("h")) else None,
            "proto_scores": (
                extras.get("proto_scores").detach().cpu()
                if torch.is_tensor(extras.get("proto_scores"))
                else None
            ),
            "target_proto_probs": (
                extras.get("target_proto_probs").detach().cpu()
                if torch.is_tensor(extras.get("target_proto_probs"))
                else None
            ),
            "proto_positive_scores": (
                extras.get("proto_positive_scores").detach().cpu()
                if torch.is_tensor(extras.get("proto_positive_scores"))
                else None
            ),
            "proto_negative_scores": (
                extras.get("proto_negative_scores").detach().cpu()
                if torch.is_tensor(extras.get("proto_negative_scores"))
                else None
            ),
            "head_key": str(extras.get("head_key", "default")),
        }
        stage_records.append(record)

    def build_feature_map(self) -> List[dict]:
        return [item.to_dict() for item in self.feature_meta]

    def _concat(self, stage: str, key: str) -> torch.Tensor | None:
        records = self._stage_records.get(stage, [])
        tensors = [item[key] for item in records if torch.is_tensor(item.get(key))]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _prototype_cards(self, stage: str, prototype_head) -> List[dict]:
        records = self._stage_records.get(stage, [])
        if not records:
            return []

        prototype_cards: List[dict] = []
        for head_key in prototype_head.head_keys:
            buffers = prototype_head._get_buffers(head_key)
            prototypes = buffers[0]
            counts = buffers[3] if len(buffers) >= 6 else buffers[1]
            prototypes_cpu = prototypes.detach().cpu()
            health = prototype_head.export_health(head_key)
            for class_id in range(prototypes_cpu.shape[0]):
                for proto_id in range(prototypes_cpu.shape[1]):
                    vec = prototypes_cpu[class_id, proto_id]
                    top_k = min(self.top_t, vec.numel())
                    top_indices = torch.topk(vec.abs(), k=top_k).indices.tolist() if top_k > 0 else []
                    prototype_cards.append(
                        {
                            "stage": stage,
                            "head_key": head_key,
                            "class_id": int(class_id),
                            "prototype_id": int(proto_id),
                            "prototype_vector": vec.tolist(),
                            "top_dims": [
                                {
                                    "dimension_index": int(dim_index),
                                    "value": float(vec[int(dim_index)].item()),
                                }
                                for dim_index in top_indices
                            ],
                            "occupancy_metrics": {
                                "assignment_mass": float(counts[class_id, proto_id].detach().cpu().item()),
                                "assignment_ratio": float(health["max_assignment_ratio"][class_id]),
                            },
                            "complementarity_metrics": {
                                "max_same_class_cos": float(health["max_offdiag_cos"][class_id]),
                            },
                        }
                    )
        return prototype_cards

    def build_stage_payload(
        self,
        stage: str,
        prototype_head,
        variant_id: str = "baseline",
        active_components: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        file_ids = self._concat(stage, "file_ids")
        alpha_t = self._concat(stage, "alpha_t")
        alpha_f = self._concat(stage, "alpha_f")
        s = self._concat(stage, "S")
        w_t = self._concat(stage, "w_t")
        w_f = self._concat(stage, "w_f")
        g_t = self._concat(stage, "g_t")
        g_f = self._concat(stage, "g_f")
        h = self._concat(stage, "h")
        a_t = next((item["A_t"] for item in self._stage_records.get(stage, []) if torch.is_tensor(item["A_t"])), None)
        a_f = next((item["A_f"] for item in self._stage_records.get(stage, []) if torch.is_tensor(item["A_f"])), None)
        prototype_health = prototype_head.export_all_health() if prototype_head is not None else None

        payload: Dict[str, object] = {
            "feature_map": self.build_feature_map(),
            "variant_id": str(variant_id),
            "active_components": active_components or {},
            "active_feature_count": int(len(self.feature_meta)),
            "alpha_time_report": build_alpha_report(file_ids, alpha_t, "patch_channel_weights"),
            "alpha_freq_report": build_alpha_report(file_ids, alpha_f, "band_channel_weights"),
            "role_basis_time": build_role_basis_report(
                build_basis_labels(self.time_operator_names, self.time_indicator_names, "time"),
                a_t,
                "time_role",
            ),
            "role_basis_freq": build_role_basis_report(
                build_basis_labels(self.freq_operator_names, self.freq_indicator_names, "freq"),
                a_f,
                "freq_role",
            ),
            "patch_band_similarity": build_patch_band_report(file_ids, s, w_t, w_f),
            "compact_concept_report": build_compact_concept_report(file_ids, h, g_t, g_f, self.role_dim),
            "prototype_cards": self._prototype_cards(stage, prototype_head) if prototype_head is not None else None,
            "prototype_health": prototype_health,
        }
        return payload
