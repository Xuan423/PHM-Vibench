"""Stage2 evaluation protocol utilities.

This module implements the test-episodic (test → support+query) split used by
TSPN_CL Stage2 K-shot adaptation, with strict guarantees:
- The support set is sampled once (seeded) and then frozen for the whole run.
- Stage2 parameter updates may use support only (never query).

The split is expressed in terms of dataset indices (0..len(dataset)-1) so it can
be applied via ``torch.utils.data.Subset``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SupportQuerySplit:
    protocol: str
    seed_used: int
    k_shot: int
    support_indices_by_class: Dict[int, List[int]]
    support_indices: List[int]
    query_indices: List[int]
    counts_by_class: Dict[int, Dict[str, int]]


def _to_int_label(y: Any) -> int:
    """Best-effort conversion of a label to int."""
    try:
        # torch scalar / numpy scalar
        if hasattr(y, "item"):
            return int(y.item())
    except Exception:
        pass
    try:
        return int(y)
    except Exception as exc:
        raise TypeError(f"Unsupported label type: {type(y)}") from exc


def collect_label_indices(dataset: Any) -> Dict[int, List[int]]:
    """Collect dataset indices grouped by class label.

    Notes
    -----
    This iterates over the dataset and reads ``sample['y']``.
    """
    label_to_indices: Dict[int, List[int]] = {}
    n = len(dataset)
    for idx in range(n):
        sample = dataset[idx]
        if not isinstance(sample, Mapping) or "y" not in sample:
            raise ValueError("Dataset samples must be mappings that contain key 'y'.")
        lbl = _to_int_label(sample["y"])
        label_to_indices.setdefault(lbl, []).append(int(idx))
    return label_to_indices


def build_support_query_split(
    dataset: Any,
    *,
    k_shot: int,
    seed: int,
    protocol: str = "test_episodic",
) -> SupportQuerySplit:
    """Build a seeded per-class K-shot support/query split (without replacement)."""
    if k_shot <= 0:
        raise ValueError(f"k_shot must be > 0, got {k_shot}")

    label_to_indices = collect_label_indices(dataset)
    rng = np.random.default_rng(int(seed))

    support_by_class: Dict[int, List[int]] = {}
    counts_by_class: Dict[int, Dict[str, int]] = {}

    support_flat: List[int] = []
    all_indices = set(range(len(dataset)))

    for lbl in sorted(label_to_indices.keys()):
        indices = list(label_to_indices[lbl])
        if len(indices) < k_shot:
            # Not enough samples for this class; return an empty support set for it.
            support_by_class[lbl] = []
            counts_by_class[lbl] = {"available": len(indices), "support": 0, "query": len(indices)}
            continue

        # Deterministic sampling without replacement.
        chosen = rng.choice(np.array(indices, dtype=np.int64), size=int(k_shot), replace=False).tolist()
        chosen = [int(x) for x in chosen]
        support_by_class[lbl] = sorted(chosen)
        support_flat.extend(chosen)

        counts_by_class[lbl] = {
            "available": int(len(indices)),
            "support": int(k_shot),
            "query": int(len(indices) - k_shot),
        }

    support_set = set(support_flat)
    query_indices = sorted(list(all_indices - support_set))

    return SupportQuerySplit(
        protocol=str(protocol),
        seed_used=int(seed),
        k_shot=int(k_shot),
        support_indices_by_class=support_by_class,
        support_indices=sorted(list(support_set)),
        query_indices=query_indices,
        counts_by_class=counts_by_class,
    )


def split_has_enough_support(split: SupportQuerySplit) -> bool:
    """Return True if every class has at least one support sample."""
    if not split.support_indices:
        return False
    return all(len(v) > 0 for v in split.support_indices_by_class.values())


def write_split_json(split: SupportQuerySplit, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(split), f, indent=2, ensure_ascii=False)


def validate_split(split: SupportQuerySplit, dataset_len: int) -> Tuple[bool, List[str]]:
    """Validate basic invariants (disjoint, index range, counts)."""
    issues: List[str] = []
    sup = set(split.support_indices)
    qry = set(split.query_indices)
    if sup & qry:
        issues.append("support/query are not disjoint")
    if len(sup) + len(qry) != dataset_len:
        issues.append("support+query do not cover the dataset length")
    if any((i < 0 or i >= dataset_len) for i in sup.union(qry)):
        issues.append("some indices are out of range")
    return (len(issues) == 0), issues

