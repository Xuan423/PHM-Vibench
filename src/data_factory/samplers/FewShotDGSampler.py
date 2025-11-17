"""Few-shot domain generalization sampler utilities."""

from __future__ import annotations

import math
import multiprocessing as mp
import warnings
from collections import defaultdict
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Dict, Iterator, List, Optional, Tuple, Union
from torch.utils.data import Sampler

from ..dataset_task.Dataset_cluster import IdIncludedDataset
from ..batch_sync import ChunkedBatchIndex, EpisodeChunkTracker


@dataclass
class EpisodeChunkSpec:
    indices: List[Union[int, ChunkedBatchIndex]]
    layout: EpisodeLayout
    chunk_uid: Optional[str] = None


@dataclass
class EpisodeLabelLayout:
    system_id: str
    domain_id: str
    label_id: str
    support_start: int
    support_count: int
    query_start: int
    query_count: int


@dataclass
class EpisodeLayout:
    size: int
    requested_support: int
    requested_query: int
    labels: List[EpisodeLabelLayout]
    episode_id: Optional[str] = None
    chunk_index: int = 0
    chunk_count: int = 1
    chunk_uid: Optional[str] = None


class FewShotDGSampler(Sampler[List[int]]):
    """Episode-based sampler that preserves domain labels for few-shot DG.

    Each iteration yields the global indices that form one episode consisting of
    support and query samples grouped by system → domain → label. The sampler
    gracefully degrades when the requested shot counts exceed the available
    examples for a label by reducing the counts to the maximum feasible number
    and emitting a warning once per label.
    """

    def __init__(
        self,
        dataset: IdIncludedDataset,
        few_shot_cfg,
        *,
        mode: str = "train",
        default_seed: int = 0,
        iteration_batch_size: Optional[int] = None,
        chunk_tracker: Optional[EpisodeChunkTracker] = None,
        stage_mode: Optional[str] = None,
    ) -> None:
        if not isinstance(dataset, IdIncludedDataset):
            raise ValueError("FewShotDGSampler expects an IdIncludedDataset instance")
        if not hasattr(dataset, "metadata") or dataset.metadata is None:
            raise ValueError("FewShotDGSampler requires dataset.metadata to be populated")
        if few_shot_cfg is None or not getattr(few_shot_cfg, "enabled", False):
            raise ValueError("few_shot configuration must be provided with enabled=True")

        self.dataset = dataset
        self.mode = mode
        self.cfg = few_shot_cfg
        self.iteration_batch_size = max(int(iteration_batch_size or 0), 0)
        self._chunk_tracker = chunk_tracker
        self.stage_mode = stage_mode

        self.system_key = getattr(few_shot_cfg, "system_key", "Dataset_id")
        self.domain_key = getattr(few_shot_cfg, "domain_key", "Domain_id")
        self.label_key = getattr(few_shot_cfg, "label_key", "Label")

        self.systems_per_episode = max(int(getattr(few_shot_cfg, "systems_per_episode", 1)), 1)
        self.domains_per_system = max(int(getattr(few_shot_cfg, "domains_per_episode", 1)), 1)
        self.classes_per_domain = max(int(getattr(few_shot_cfg, "classes_per_domain", 1)), 1)
        self.support_per_class = max(int(getattr(few_shot_cfg, "support_per_class", 1)), 0)
        self.query_per_class = max(int(getattr(few_shot_cfg, "query_per_class", 1)), 0)
        self.preserve_labels = bool(getattr(few_shot_cfg, "preserve_labels", True))
        self.warn_on_shortfall = bool(getattr(few_shot_cfg, "warn_on_shortfall", True))

        requested_total = self.support_per_class + self.query_per_class
        if requested_total <= 0:
            raise ValueError("support_per_class + query_per_class must be greater than zero")

        base_seed = getattr(few_shot_cfg, "seed", None)
        if base_seed is None:
            base_seed = getattr(few_shot_cfg, "random_seed", None)
        if base_seed is None:
            base_seed = default_seed
        # Offset validation so that train/val/test produce different deterministic streams.
        mode_offset = {"train": 0, "val": 113, "test": 251}.get(mode, 17)
        self.base_seed = int(base_seed) + mode_offset
        self.epoch = 0

        requested_episodes = getattr(few_shot_cfg, "episodes_per_epoch", None)
        self._user_requested_episodes = (
            requested_episodes is not None and int(requested_episodes) > 0
        )
        self.episodes_per_epoch = max(int(requested_episodes or 0), 0)

        # Pre-compute sample pools indexed by (system, domain, label).
        self._indices_by_label: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self._labels_per_domain: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._domains_per_system: Dict[str, List[str]] = defaultdict(list)

        for global_idx, mapping in enumerate(self.dataset.get_file_windows_list()):
            file_id = mapping["file_id"]
            if file_id not in self.dataset.metadata:
                continue
            meta = self.dataset.metadata[file_id]
            system_id = str(meta.get(self.system_key, "unknown"))
            domain_id = str(meta.get(self.domain_key, "unknown"))
            label_id = str(meta.get(self.label_key, "unknown"))
            key = (system_id, domain_id, label_id)
            self._indices_by_label[key].append(global_idx)

        for (system_id, domain_id, label_id), indices in self._indices_by_label.items():
            domain_key = (system_id, domain_id)
            if label_id not in self._labels_per_domain[domain_key]:
                self._labels_per_domain[domain_key].append(label_id)
            if domain_id not in self._domains_per_system[system_id]:
                self._domains_per_system[system_id].append(domain_id)
            # Sort indices for deterministic slicing when sampling without replacement.
            indices.sort()

        self._systems = sorted(self._domains_per_system.keys())
        if not self._systems:
            raise ValueError("No systems available for few-shot sampling.")

        if self.episodes_per_epoch == 0:
            # Default heuristic: cover each domain roughly once per epoch.
            total_domains = sum(len(domains) for domains in self._domains_per_system.values())
            shots_per_episode = max(self.classes_per_domain * requested_total * self.domains_per_system, 1)
            approx_samples = sum(len(v) for v in self._indices_by_label.values())
            episodes = approx_samples // shots_per_episode if shots_per_episode else 1
            self.episodes_per_epoch = max(episodes, total_domains)

        if (self.stage_mode or "").lower() == "stage2" and not self._user_requested_episodes:
            self.episodes_per_epoch = 1

        # Determine global fallback counts for support/query if needed.
        self._effective_support, self._effective_query = self._compute_effective_shots(requested_total)
        self._warned_labels: set[str] = set()
        self._layout_queue = None
        try:
            start_method = mp.get_start_method(allow_none=True) or "spawn"
            ctx = mp.get_context(start_method)
            self._layout_queue = ctx.Queue()
            cancel_join = getattr(self._layout_queue, "cancel_join_thread", None)
            if callable(cancel_join):
                cancel_join()
        except (PermissionError, OSError):
            warnings.warn(
                "FewShotDGSampler falling back to in-process queue for episode layouts.",
                RuntimeWarning,
            )
            self._layout_queue = Queue()
        self._episode_uid = 0
        self._warned_support_overflow = False
        expected_query = (
            self.systems_per_episode
            * self.domains_per_system
            * self.classes_per_domain
            * self.query_per_class
        )
        if self.iteration_batch_size > 0:
            self._estimated_chunks_per_episode = max(
                1, math.ceil(max(1, expected_query) / self.iteration_batch_size)
            )
        else:
            self._estimated_chunks_per_episode = 1

    def _compute_effective_shots(self, requested_total: int) -> Tuple[int, int]:
        min_count = math.inf
        for indices in self._indices_by_label.values():
            min_count = min(min_count, len(indices))
        if min_count == math.inf:
            raise ValueError("Unable to compute label counts for few-shot sampler.")

        support = self.support_per_class
        query = self.query_per_class

        if min_count >= requested_total:
            return support, query

        # Reduce shots globally to the feasible minimum while keeping both >0 when possible.
        effective_total = min_count
        if effective_total < 2:
            # Fall back to at least a single query if possible so evaluation can proceed.
            warnings.warn(
                "FewShotDGSampler detected labels with fewer than 2 samples; "
                "episodes may include repeated indices or reduced shots.",
                RuntimeWarning,
            )
            return max(min(effective_total, support), 0), max(min(effective_total, query), 0)

        # Split the available samples between support/query in proportion to the request.
        support_ratio = support / (support + query)
        effective_support = max(int(round(effective_total * support_ratio)), 1)
        effective_support = min(effective_support, effective_total - 1)
        effective_query = effective_total - effective_support
        if self.warn_on_shortfall:
            warnings.warn(
                "FewShotDGSampler reduced shots from "
                f"{support}+{query} to {effective_support}+{effective_query} "
                "due to limited examples in at least one label.",
                RuntimeWarning,
            )
        return effective_support, effective_query

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic but shuffled sampling across epochs."""
        self.epoch = int(epoch)
        self._episode_uid = 0

    def __len__(self) -> int:
        return self.episodes_per_epoch * self._estimated_chunks_per_episode

    def pop_layout(self) -> Optional[EpisodeLayout]:
        """Return the next episode layout emitted during iteration."""
        try:
            return self._layout_queue.get_nowait()
        except Empty:
            return None

    def __iter__(self) -> Iterator[List[int]]:
        import random

        rng = random.Random(self.base_seed + self.epoch)
        systems = list(self._systems)

        for _ in range(self.episodes_per_epoch):
            rng.shuffle(systems)
            chosen_systems = systems[: min(self.systems_per_episode, len(systems))]
            episode_indices: List[int] = []
            episode_layout_labels: List[EpisodeLabelLayout] = []

            if len(chosen_systems) < self.systems_per_episode and self.warn_on_shortfall:
                warnings.warn(
                    "Requested systems_per_episode exceeds available systems; "
                    "sampling with replacement.",
                    RuntimeWarning,
                )
                chosen_systems = [rng.choice(systems) for _ in range(self.systems_per_episode)]

            for system_id in chosen_systems:
                domains = list(self._domains_per_system.get(system_id, []))
                if not domains:
                    continue
                rng.shuffle(domains)
                selected_domains = domains[: min(self.domains_per_system, len(domains))]
                if len(selected_domains) < self.domains_per_system and self.warn_on_shortfall:
                    warnings.warn(
                        f"System {system_id} has fewer domains than requested; allowing repeats.",
                        RuntimeWarning,
                    )
                    selected_domains = [rng.choice(domains) for _ in range(self.domains_per_system)]

                for domain_id in selected_domains:
                    domain_key = (system_id, domain_id)
                    labels = list(self._labels_per_domain.get(domain_key, []))
                    if not labels:
                        continue
                    rng.shuffle(labels)
                    selected_labels = labels[: min(self.classes_per_domain, len(labels))]
                    if len(selected_labels) < self.classes_per_domain and self.warn_on_shortfall:
                        warnings.warn(
                            f"Domain {domain_id} in system {system_id} has fewer labels than requested; "
                            "reusing labels.",
                            RuntimeWarning,
                        )
                        selected_labels = [rng.choice(labels) for _ in range(self.classes_per_domain)]

                    for label_id in selected_labels:
                        indices = self._indices_by_label.get((system_id, domain_id, label_id), [])
                        if not indices:
                            continue
                        support_indices, query_indices = self._draw_indices_for_label(
                            indices,
                            label_id,
                            rng,
                        )
                        if not support_indices and not query_indices:
                            continue

                        support_start = len(episode_indices)
                        if support_indices:
                            episode_indices.extend(support_indices)
                        query_start = len(episode_indices)
                        if query_indices:
                            episode_indices.extend(query_indices)

                        episode_layout_labels.append(
                            EpisodeLabelLayout(
                                system_id=system_id,
                                domain_id=domain_id,
                                label_id=label_id,
                                support_start=support_start,
                                support_count=len(support_indices),
                                query_start=query_start,
                                query_count=len(query_indices),
                            )
                        )

            if episode_indices:
                episode_id = self._next_episode_id()
                chunk_specs = self._chunk_episode(
                    episode_indices,
                    episode_layout_labels,
                    episode_id=episode_id,
                )
                for spec in chunk_specs:
                    if self._layout_queue is not None:
                        self._layout_queue.put(spec.layout)
                    yield spec.indices

    @property
    def layout_queue(self):
        """Expose the multiprocessing queue carrying episode layouts."""
        return self._layout_queue

    @property
    def chunk_tracker(self) -> Optional[EpisodeChunkTracker]:
        return self._chunk_tracker

    def _draw_indices_for_label(
        self,
        candidates: List[int],
        label_id: str,
        rng,
    ) -> Tuple[List[int], List[int]]:
        available = len(candidates)
        support = self._effective_support
        query = self._effective_query
        required = support + query

        if available >= required:
            # Sample without replacement for deterministic reproducibility.
            chosen = rng.sample(candidates, required)
            support_indices = chosen[:support]
            query_indices = chosen[support:]
            return support_indices, query_indices

        if available == 0:
            return [], []

        if self.warn_on_shortfall and label_id not in self._warned_labels:
            warnings.warn(
                f"Label {label_id} only has {available} samples; returning all available indices.",
                RuntimeWarning,
            )
            self._warned_labels.add(label_id)
        pool = list(candidates)
        rng.shuffle(pool)
        support_taken = min(support, len(pool))
        support_indices = pool[:support_taken]
        remaining = pool[support_taken:]
        query_taken = min(query, len(remaining))
        query_indices = remaining[:query_taken]
        return support_indices, query_indices

    def _next_episode_id(self) -> str:
        uid = self._episode_uid
        self._episode_uid += 1
        return f"{self.mode}-epoch{self.epoch}-ep{uid}"

    def _chunk_episode(
        self,
        episode_indices: List[int],
        labels: List[EpisodeLabelLayout],
        *,
        episode_id: str,
    ) -> List[EpisodeChunkSpec]:
        total_query = sum(label.query_count for label in labels)
        support_total = sum(label.support_count for label in labels)
        chunk_limit = self.iteration_batch_size
        if chunk_limit <= 0 or total_query <= chunk_limit:
            layout = EpisodeLayout(
                size=len(episode_indices),
                requested_support=self._effective_support,
                requested_query=self._effective_query,
                labels=labels,
                episode_id=episode_id,
                chunk_index=0,
                chunk_count=1,
            )
            return [EpisodeChunkSpec(indices=episode_indices, layout=layout)]

        chunk_limit = max(1, chunk_limit)
        if (
            support_total >= chunk_limit
            and not self._warned_support_overflow
        ):
            warnings.warn(
                "Configured batch_size is smaller than the support set; "
                "each iteration will include the full support set in addition "
                "to the limited query samples.",
                RuntimeWarning,
            )
            self._warned_support_overflow = True
        progress = [0 for _ in labels]
        remaining = total_query
        chunk_maps: List[Dict[int, Tuple[int, int]]] = []

        while remaining > 0:
            budget = chunk_limit
            current_map: Dict[int, Tuple[int, int]] = {}
            for idx, label in enumerate(labels):
                if budget <= 0:
                    break
                available = label.query_count - progress[idx]
                if available <= 0:
                    continue
                take = min(available, budget)
                current_map[idx] = (progress[idx], take)
                progress[idx] += take
                budget -= take
                remaining -= take
                if budget == 0:
                    break
            if not current_map:
                # No more queries can be assigned (should not happen but guard anyway).
                break
            chunk_maps.append(current_map)

        if not chunk_maps:
            # Fallback to original single-chunk behaviour.
            layout = EpisodeLayout(
                size=len(episode_indices),
                requested_support=self._effective_support,
                requested_query=self._effective_query,
                labels=labels,
                episode_id=episode_id,
                chunk_index=0,
                chunk_count=1,
            )
            return [EpisodeChunkSpec(indices=episode_indices, layout=layout)]

        emitted_maps = chunk_maps

        chunk_count = len(emitted_maps)
        specs: List[EpisodeChunkSpec] = []
        for chunk_index, chunk_map in enumerate(emitted_maps):
            chunk_indices: List[int] = []
            layout_labels: List[EpisodeLabelLayout] = []
            cursor = 0
            for label_idx, label in enumerate(labels):
                support_slice = episode_indices[
                    label.support_start : label.support_start + label.support_count
                ]
                support_count = len(support_slice)
                if support_slice:
                    chunk_indices.extend(support_slice)

                query_offset, query_take = chunk_map.get(label_idx, (0, 0))
                query_slice = []
                if query_take > 0:
                    start = label.query_start + query_offset
                    end = start + query_take
                    query_slice = episode_indices[start:end]
                    chunk_indices.extend(query_slice)

                query_count = len(query_slice)
                layout_labels.append(
                    EpisodeLabelLayout(
                        system_id=label.system_id,
                        domain_id=label.domain_id,
                        label_id=label.label_id,
                        support_start=cursor,
                        support_count=support_count,
                        query_start=cursor + support_count,
                        query_count=query_count,
                    )
                )
                cursor += support_count + query_count

            layout = EpisodeLayout(
                size=cursor,
                requested_support=self._effective_support,
                requested_query=self._effective_query,
                labels=layout_labels,
                episode_id=episode_id,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )
            chunk_uid: Optional[str] = None
            if self._chunk_tracker is not None:
                handle = self._chunk_tracker.register_layout(layout)
                chunk_uid = handle.chunk_uid
                layout.chunk_uid = chunk_uid

            wrapped_indices: List[Union[int, ChunkedBatchIndex]]
            if chunk_uid:
                wrapped_indices = [
                    ChunkedBatchIndex(idx=sample_idx, chunk_uid=chunk_uid, chunk_seq=seq)
                    for seq, sample_idx in enumerate(chunk_indices)
                ]
            else:
                wrapped_indices = chunk_indices

            specs.append(EpisodeChunkSpec(indices=wrapped_indices, layout=layout, chunk_uid=chunk_uid))

        return specs
