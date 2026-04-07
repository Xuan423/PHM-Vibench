from __future__ import annotations

from typing import Iterable, List

import torch

from .feature_schema import build_compact_concept_labels


def tensor_to_nested_list(tensor: torch.Tensor | None) -> list | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def build_alpha_report(
    file_ids: torch.Tensor | None,
    weights: torch.Tensor | None,
    axis_name: str,
) -> List[dict]:
    if file_ids is None or weights is None:
        return []
    file_ids_cpu = file_ids.detach().cpu().view(-1).tolist()
    weights_cpu = weights.detach().cpu()
    report = []
    for index, file_id in enumerate(file_ids_cpu):
        report.append(
            {
                "file_id": int(file_id),
                axis_name: weights_cpu[index].tolist(),
            }
        )
    return report


def build_role_basis_report(labels: Iterable[str], basis: torch.Tensor | None, role_prefix: str) -> List[dict]:
    if basis is None:
        return []
    basis_cpu = basis.detach().cpu()
    labels_list = list(labels)
    report: List[dict] = []
    for role_idx in range(basis_cpu.shape[1]):
        weights = basis_cpu[:, role_idx]
        top_k = min(8, weights.numel())
        top_indices = torch.topk(weights, k=top_k).indices.tolist() if top_k > 0 else []
        report.append(
            {
                "role_id": f"{role_prefix}_{role_idx}",
                "weights": weights.tolist(),
                "top_basis_items": [
                    {
                        "basis_index": int(item_index),
                        "label": labels_list[int(item_index)],
                        "weight": float(weights[int(item_index)].item()),
                    }
                    for item_index in top_indices
                ],
            }
        )
    return report


def build_patch_band_report(
    file_ids: torch.Tensor | None,
    s: torch.Tensor | None,
    w_t: torch.Tensor | None,
    w_f: torch.Tensor | None,
) -> List[dict]:
    if file_ids is None or s is None or w_t is None or w_f is None:
        return []
    file_ids_cpu = file_ids.detach().cpu().view(-1).tolist()
    s_cpu = s.detach().cpu()
    w_t_cpu = w_t.detach().cpu()
    w_f_cpu = w_f.detach().cpu()
    report = []
    for index, file_id in enumerate(file_ids_cpu):
        report.append(
            {
                "file_id": int(file_id),
                "similarity": s_cpu[index].tolist(),
                "time_weights": w_t_cpu[index].tolist(),
                "freq_weights": w_f_cpu[index].tolist(),
            }
        )
    return report


def build_compact_concept_report(
    file_ids: torch.Tensor | None,
    h: torch.Tensor | None,
    g_t: torch.Tensor | None,
    g_f: torch.Tensor | None,
    role_dim: int,
) -> List[dict]:
    if file_ids is None or h is None or g_t is None or g_f is None:
        return []
    file_ids_cpu = file_ids.detach().cpu().view(-1).tolist()
    h_cpu = h.detach().cpu()
    g_t_cpu = g_t.detach().cpu()
    g_f_cpu = g_f.detach().cpu()
    dimension_labels = build_compact_concept_labels(role_dim)
    report = []
    for index, file_id in enumerate(file_ids_cpu):
        top_k = min(8, h_cpu[index].numel())
        top_indices = torch.topk(h_cpu[index].abs(), k=top_k).indices.tolist() if top_k > 0 else []
        report.append(
            {
                "file_id": int(file_id),
                "g_t": g_t_cpu[index].tolist(),
                "g_f": g_f_cpu[index].tolist(),
                "h": h_cpu[index].tolist(),
                "top_compact_dims": [
                    {
                        "dimension_index": int(dim_index),
                        "label": dimension_labels[int(dim_index)],
                        "value": float(h_cpu[index, int(dim_index)].item()),
                    }
                    for dim_index in top_indices
                ],
            }
        )
    return report
