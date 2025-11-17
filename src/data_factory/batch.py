"""Batch utilities for episode-aware few-shot sampling."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data._utils.collate import default_collate

from .batch_sync import ChunkHandle, EpisodeChunkTracker
from .samplers.FewShotDGSampler import EpisodeLayout

LOGGER = logging.getLogger(__name__)


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
    episode_id: Optional[str] = None
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_uid: Optional[str] = None

    @property
    def is_first_chunk(self) -> bool:
        return self.chunk_index == 0

    @property
    def is_last_chunk(self) -> bool:
        return self.chunk_index >= max(self.chunk_count - 1, 0)


class EpisodeCollate:
    """Collate function that materialises support/query tensors per episode."""

    def __init__(
        self,
        *,
        layout_queue: Optional[Any] = None,
        metadata: Optional[Any] = None,
        layout_tracker: Optional[EpisodeChunkTracker] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._layout_queue = layout_queue
        self.metadata = metadata
        self._layout_tracker = layout_tracker
        self._logger = logger or LOGGER

    # ------------------------------------------------------------------
    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Any:
        chunk_uid = self._extract_chunk_uid(samples)
        layout, handle = self._resolve_layout(chunk_uid)

        if layout is None:
            return default_collate(samples)
        if layout.size != len(samples):
            message = (
                "Episode layout size mismatch detected "
                f"(chunk_uid={chunk_uid or getattr(layout, 'chunk_uid', 'n/a')}, "
                f"expected={layout.size}, received={len(samples)})"
            )
            self._logger.error(message)
            raise RuntimeError(message)

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

        batch = EpisodeBatch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            support_file_ids=support_file_ids,
            query_file_ids=query_file_ids,
            label_views=label_views,
            layout=layout,
            flat_batch=flat_batch,
            episode_id=getattr(layout, "episode_id", None),
            chunk_index=getattr(layout, "chunk_index", 0),
            chunk_count=getattr(layout, "chunk_count", 1),
            chunk_uid=chunk_uid or getattr(layout, "chunk_uid", None),
        )
        if batch.chunk_uid:
            batch.flat_batch["chunk_uid"] = batch.chunk_uid
            batch.flat_batch["__chunk_uid"] = batch.chunk_uid
        if batch.episode_id:
            batch.flat_batch["episode_id"] = batch.episode_id
        if handle is not None and self._layout_tracker is not None:
            self._layout_tracker.report_consumed(handle.chunk_uid)
        return batch

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

    # ------------------------------------------------------------------
    def _resolve_layout(
        self,
        chunk_uid: Optional[str],
    ) -> Tuple[Optional[EpisodeLayout], Optional[ChunkHandle]]:
        start = time.perf_counter()
        handle: Optional[ChunkHandle] = None
        layout: Optional[EpisodeLayout] = None
        if self._layout_tracker is not None and chunk_uid:
            handle = self._layout_tracker.claim_layout(chunk_uid)
            if handle is None:
                message = f"Timed out waiting for chunk layout (chunk_uid={chunk_uid})"
                self._logger.error(message)
                raise RuntimeError(message)
            layout = handle.layout
        else:
            layout = self._pop_layout_from_queue()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if chunk_uid and layout is not None:
            self._logger.debug(
                "Resolved chunk %s layout in %.3f ms (tracker=%s)",
                chunk_uid,
                elapsed_ms,
                bool(self._layout_tracker),
            )
        return layout, handle

    def _pop_layout_from_queue(self) -> Optional[EpisodeLayout]:
        if self._layout_queue is None:
            return None
        try:
            return self._layout_queue.get_nowait()
        except Empty:
            return None

    @staticmethod
    def _extract_chunk_uid(samples: Sequence[Dict[str, Any]]) -> Optional[str]:
        chunk_uids = {
            sample.get("__chunk_uid")
            for sample in samples
            if isinstance(sample, dict) and sample.get("__chunk_uid")
        }
        chunk_uids.discard(None)
        if not chunk_uids:
            return None
        if len(chunk_uids) > 1:
            raise RuntimeError(f"EpisodeCollate detected mixed chunk_uids: {sorted(chunk_uids)}")
        return next(iter(chunk_uids))


__all__ = [
    "EpisodeLabelView",
    "EpisodeBatch",
    "EpisodeCollate",
]
