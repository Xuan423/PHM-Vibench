"""Batch utilities for episode-aware few-shot sampling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data._utils.collate import default_collate

from .samplers.FewShotDGSampler import EpisodeLayout


@dataclass
class EpisodeLabelView:
    """View of support/query slices for a single label inside an episode."""

    system_id: str
    domain_id: str
    label_id: str
    local_label: int
    support_slice: slice
    query_slice: slice
    support_count: int
    query_count: int
    original_label: Optional[Any] = None


@dataclass
class EpisodeBatch:
    """Structured batch separating support/query tensors and metadata."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    support_file_ids: List[Any]
    query_file_ids: List[Any]
    label_views: List[EpisodeLabelView]
    layout: EpisodeLayout
    flat_batch: Dict[str, Any] = field(default_factory=dict)


class EpisodeCollate:
    """Collate function that materialises support/query tensors per episode."""

    def __init__(self, *, layout_queue: Optional[Any] = None, metadata: Optional[Any] = None) -> None:
        self._layout_queue = layout_queue
        self.metadata = metadata

    # ------------------------------------------------------------------
    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Any:
        layout: Optional[EpisodeLayout] = None
        if self._layout_queue is not None:
            try:
                layout = self._layout_queue.get_nowait()
            except Empty:
                layout = None
        if layout is None:
            return default_collate(samples)
        if layout.size != len(samples):
            warnings.warn(
                "Episode layout size mismatch; falling back to default collate.",
                RuntimeWarning,
            )
            return default_collate(samples)

        flat_batch = default_collate(samples)
        if not isinstance(flat_batch, dict) or "x" not in flat_batch or "y" not in flat_batch:
            return flat_batch

        x_tensor = flat_batch["x"]
        y_tensor = flat_batch["y"]
        file_ids = [sample.get("file_id") for sample in samples]

        support_chunks: List[torch.Tensor] = []
        query_chunks: List[torch.Tensor] = []
        support_y_chunks: List[torch.Tensor] = []
        query_y_chunks: List[torch.Tensor] = []
        support_file_ids: List[Any] = []
        query_file_ids: List[Any] = []
        label_views: List[EpisodeLabelView] = []

        support_offset = 0
        query_offset = 0

        for local_idx, label in enumerate(layout.labels):
            support_start = label.support_start
            support_end = support_start + label.support_count
            query_start = label.query_start
            query_end = query_start + label.query_count

            support_slice_tensor = x_tensor[support_start:support_end]
            support_slice_labels = y_tensor[support_start:support_end]
            query_slice_tensor = x_tensor[query_start:query_end]
            query_slice_labels = y_tensor[query_start:query_end]

            if label.support_count:
                support_chunks.append(support_slice_tensor)
                support_y_chunks.append(support_slice_labels)
                support_file_ids.extend(file_ids[support_start:support_end])
            if label.query_count:
                query_chunks.append(query_slice_tensor)
                query_y_chunks.append(query_slice_labels)
                query_file_ids.extend(file_ids[query_start:query_end])

            support_slice = slice(support_offset, support_offset + label.support_count)
            query_slice = slice(query_offset, query_offset + label.query_count)
            support_offset += label.support_count
            query_offset += label.query_count

            original_label_value: Optional[Any] = None
            if label.support_count:
                original_label_value = support_slice_labels[0].item() if hasattr(support_slice_labels, "numel") and support_slice_labels.numel() else None
            elif label.query_count:
                original_label_value = query_slice_labels[0].item() if hasattr(query_slice_labels, "numel") and query_slice_labels.numel() else None

            label_views.append(
                EpisodeLabelView(
                    system_id=label.system_id,
                    domain_id=label.domain_id,
                    label_id=label.label_id,
                    local_label=local_idx,
                    support_slice=support_slice,
                    query_slice=query_slice,
                    support_count=label.support_count,
                    query_count=label.query_count,
                    original_label=original_label_value,
                )
            )

        support_x = self._stack_or_empty(support_chunks, x_tensor)
        support_y = self._stack_or_empty(support_y_chunks, y_tensor)
        query_x = self._stack_or_empty(query_chunks, x_tensor)
        query_y = self._stack_or_empty(query_y_chunks, y_tensor)

        return EpisodeBatch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            support_file_ids=support_file_ids,
            query_file_ids=query_file_ids,
            label_views=label_views,
            layout=layout,
            flat_batch=flat_batch,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stack_or_empty(chunks: Iterable[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        chunks = list(chunks)
        if not chunks:
            shape = (0,) + tuple(reference.shape[1:])
            return reference.new_empty(shape)
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)


__all__ = [
    "EpisodeLabelView",
    "EpisodeBatch",
    "EpisodeCollate",
]
