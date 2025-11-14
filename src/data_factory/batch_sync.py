"""Chunk tracking primitives to keep Few-Shot episodes aligned across workers."""

from __future__ import annotations

import atexit
import logging
import os
import pickle
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .samplers.FewShotDGSampler import EpisodeLayout
else:  # pragma: no cover - runtime fallback to avoid circular import
    EpisodeLayout = Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkedBatchIndex:
    """Wrapper around a dataset index that carries chunk metadata."""

    idx: int
    chunk_uid: Optional[str]
    chunk_seq: int = 0

    def unwrap(self) -> int:
        """Return the underlying integer index."""
        return self.idx


@dataclass
class ChunkHandle:
    """Metadata describing a registered chunk/layout."""

    chunk_uid: str
    episode_id: Optional[str]
    chunk_index: int
    chunk_count: int
    size: int
    registered_at: float
    layout: EpisodeLayout


@dataclass
class ChunkTrackerMetrics:
    """Aggregated counters surfaced for diagnostics."""

    registered: int = 0
    claimed: int = 0
    consumed: int = 0
    timeouts: int = 0
    stale: int = 0

    @classmethod
    def from_mapping(cls, values: Dict[str, int]) -> "ChunkTrackerMetrics":
        return cls(
            registered=int(values.get("registered", 0)),
            claimed=int(values.get("claimed", 0)),
            consumed=int(values.get("consumed", 0)),
            timeouts=int(values.get("timeouts", 0)),
            stale=int(values.get("stale", 0)),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "registered": self.registered,
            "claimed": self.claimed,
            "consumed": self.consumed,
            "timeouts": self.timeouts,
            "stale": self.stale,
        }


def make_chunk_uid(
    *,
    episode_id: Optional[str],
    chunk_index: int,
    chunk_count: int,
    ordinal: int,
) -> str:
    """Return a human-readable, globally unique chunk identifier."""

    base = episode_id or "episode"
    return f"{base}#chunk{chunk_index:04d}_of_{chunk_count:04d}_{ordinal:08x}"


class EpisodeChunkTracker:
    """Durable storage that maps chunk IDs back to EpisodeLayouts."""

    def __init__(
        self,
        *,
        max_stale_seconds: float = 600.0,
        claim_poll_interval_ms: float = 2.0,
        default_timeout_ms: int = 2000,
        db_path: Optional[str] = None,
    ) -> None:
        self.max_stale_seconds = float(max_stale_seconds)
        self.claim_poll_interval_ms = max(float(claim_poll_interval_ms), 0.5)
        self.default_timeout_ms = max(int(default_timeout_ms), 0)
        self._owns_dir = db_path is None
        self._db_dir = tempfile.mkdtemp(prefix="chunk_tracker_") if self._owns_dir else None
        self._db_path = db_path or os.path.join(self._db_dir or tempfile.gettempdir(), f"chunk_tracker_{uuid.uuid4().hex}.sqlite3")
        self._conn: Optional[sqlite3.Connection] = None
        self._pid: Optional[int] = None
        self._schema_ready = False
        self._thread_lock = threading.RLock()
        self._closed = False
        self._global_ordinal = 0
        self._registrations_since_purge = 0
        self._purge_interval = 512
        self._ensure_connection()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_pid"] = None
        state["_schema_ready"] = False
        state["_thread_lock"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._thread_lock = threading.RLock()
        self._conn = None
        self._pid = None
        self._schema_ready = False

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close open connections and remove temp files."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._conn is not None:
                self._conn.close()
        finally:
            self._conn = None
        if self._owns_dir and self._db_dir and os.path.isdir(self._db_dir):
            shutil.rmtree(self._db_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def register_layout(self, layout: EpisodeLayout) -> ChunkHandle:
        """Persist a layout and return its handle."""

        with self._thread_lock:
            chunk_uid = make_chunk_uid(
                episode_id=getattr(layout, "episode_id", None),
                chunk_index=getattr(layout, "chunk_index", 0),
                chunk_count=getattr(layout, "chunk_count", 1),
                ordinal=self._global_ordinal,
            )
            self._global_ordinal += 1

        payload = pickle.dumps(layout, protocol=pickle.HIGHEST_PROTOCOL)
        now = time.time()
        conn = self._ensure_connection()
        with self._thread_lock:
            self._registrations_since_purge += 1
            if self._registrations_since_purge >= self._purge_interval:
                self._purge_stale_locked(conn, now)
                self._registrations_since_purge = 0
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO layouts
                        (chunk_uid, episode_id, chunk_index, chunk_count, size, registered_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_uid,
                        getattr(layout, "episode_id", None),
                        getattr(layout, "chunk_index", 0),
                        getattr(layout, "chunk_count", 1),
                        getattr(layout, "size", 0),
                        now,
                        sqlite3.Binary(payload),
                    ),
                )
                self._increment_metric("registered", conn)

        LOGGER.debug(
            "Registered chunk %s (episode=%s, chunk=%s/%s, size=%s)",
            chunk_uid,
            getattr(layout, "episode_id", None),
            getattr(layout, "chunk_index", 0),
            getattr(layout, "chunk_count", 1),
            getattr(layout, "size", 0),
        )
        return ChunkHandle(
            chunk_uid=chunk_uid,
            episode_id=getattr(layout, "episode_id", None),
            chunk_index=getattr(layout, "chunk_index", 0),
            chunk_count=getattr(layout, "chunk_count", 1),
            size=getattr(layout, "size", 0),
            registered_at=now,
            layout=layout,
        )

    # ------------------------------------------------------------------
    def claim_layout(self, chunk_uid: Optional[str], *, timeout_ms: Optional[int] = None) -> Optional[ChunkHandle]:
        """Fetch and remove the layout matching ``chunk_uid``."""

        if not chunk_uid:
            return None
        timeout_ms = self.default_timeout_ms if timeout_ms is None else int(timeout_ms)
        poll_interval = self.claim_poll_interval_ms / 1000.0
        deadline: Optional[float] = None
        if timeout_ms > 0:
            deadline = time.monotonic() + timeout_ms / 1000.0

        conn = self._ensure_connection()
        while True:
            row = conn.execute(
                """
                SELECT episode_id, chunk_index, chunk_count, size, registered_at, payload
                FROM layouts
                WHERE chunk_uid = ?
                """,
                (chunk_uid,),
            ).fetchone()
            if row:
                payload: bytes = bytes(row["payload"])
                layout = pickle.loads(payload)
                handle = ChunkHandle(
                    chunk_uid=chunk_uid,
                    episode_id=row["episode_id"],
                    chunk_index=row["chunk_index"],
                    chunk_count=row["chunk_count"],
                    size=row["size"],
                    registered_at=row["registered_at"],
                    layout=layout,
                )
                with conn:
                    conn.execute("DELETE FROM layouts WHERE chunk_uid = ?", (chunk_uid,))
                    self._increment_metric("claimed", conn)
                LOGGER.debug("Claimed chunk %s", chunk_uid)
                return handle

            if deadline is not None and time.monotonic() >= deadline:
                with conn:
                    self._increment_metric("timeouts", conn)
                LOGGER.warning("Timeout waiting for chunk %s layout", chunk_uid)
                return None
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    def report_consumed(self, chunk_uid: Optional[str]) -> None:
        if not chunk_uid:
            return
        conn = self._ensure_connection()
        with conn:
            self._increment_metric("consumed", conn)

    # ------------------------------------------------------------------
    def purge_stale(self) -> int:
        """Remove expired layouts and return purged count."""

        conn = self._ensure_connection()
        with self._thread_lock:
            return self._purge_stale_locked(conn, time.time())

    # ------------------------------------------------------------------
    def metrics(self) -> ChunkTrackerMetrics:
        """Return the current tracker metrics."""

        conn = self._ensure_connection()
        rows = conn.execute("SELECT name, value FROM metrics").fetchall()
        payload = {row["name"]: row["value"] for row in rows}
        return ChunkTrackerMetrics.from_mapping(payload)

    # ------------------------------------------------------------------
    def _ensure_connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._conn is not None and self._pid == pid and self._schema_ready:
            return self._conn

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = sqlite3.connect(
            self._db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._pid = pid
        self._schema_ready = False
        self._setup_connection(self._conn)
        return self._conn

    def _setup_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        self._ensure_schema(conn)
        self._schema_ready = True

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS layouts (
                chunk_uid TEXT PRIMARY KEY,
                episode_id TEXT,
                chunk_index INTEGER,
                chunk_count INTEGER,
                size INTEGER,
                registered_at REAL,
                payload BLOB NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                name TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    def _increment_metric(self, name: str, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO metrics(name, value)
            VALUES(?, 1)
            ON CONFLICT(name) DO UPDATE SET value = value + 1
            """,
            (name,),
        )

    def _purge_stale_locked(self, conn: sqlite3.Connection, now: float) -> int:
        threshold = now - self.max_stale_seconds
        with conn:
            deleted = conn.execute(
                "DELETE FROM layouts WHERE registered_at < ?",
                (threshold,),
            ).rowcount
            if deleted:
                self._increment_metric("stale", conn)
        if deleted:
            LOGGER.warning("Purged %s stale layout(s) older than %.1fs", deleted, self.max_stale_seconds)
        return deleted


__all__ = [
    "ChunkedBatchIndex",
    "ChunkHandle",
    "ChunkTrackerMetrics",
    "EpisodeChunkTracker",
    "make_chunk_uid",
]
