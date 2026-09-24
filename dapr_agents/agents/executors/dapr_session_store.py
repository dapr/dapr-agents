#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Dapr state-store-backed session transcript store.

``DaprSessionStore`` implements the Claude Agent SDK ``SessionStore``
protocol (``append`` / ``load`` plus the optional ``delete`` and
``list_subkeys``) on top of ``StateStoreService``, so a Claude session
started in one workflow activity can resume in a retried activity or on a
different pod. The protocol is duck-typed, so this module does not import
the SDK and works without the ``claude`` extra installed.

Storage layout, per session key ``{project_key}/{session_id}[/{subpath}]``:

* A *manifest* document at ``{key_prefix}{session key}`` holding the ordered
  list of chunk ids, a bounded window of recently appended entry uuids (for
  idempotent appends), the list of subpaths (main transcript only) and a
  modification time.
* Immutable *chunk* documents at ``{manifest key}#{chunk id}``, each holding
  a slice of entries no larger than ``max_chunk_bytes`` (a single oversized
  entry gets a chunk of its own).

An append writes new chunks under fresh ids first and then commits the
manifest with its ETag (first-write concurrency). A writer that loses the
race retries against the new manifest and its orphaned chunks are deleted,
so readers never observe a partial append.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dapr.clients.grpc._state import Concurrency, Consistency

from dapr_agents.storage.daprstores.stateservice import (
    StateStoreError,
    StateStoreService,
)

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1
_STATE_METADATA = {"contentType": "application/json"}
# Passed as a dict (``StateStoreService`` coerces it to ``StateOptions``) so
# each save gets its own options object.
_SAVE_OPTIONS: Dict[str, Any] = {
    "concurrency": Concurrency.first_write,
    "consistency": Consistency.strong,
}


@dataclass(frozen=True)
class DaprSessionStoreConfig:
    """
    Tuning for ``DaprSessionStore``.

    Attributes:
        key_prefix: Prefix applied to every key this store writes, on top of
            any prefix configured on the ``StateStoreService``.
        max_chunk_bytes: Upper bound for the serialized entries of one chunk
            document. Keep it below the smallest value size your state store
            accepts (for example DynamoDB caps items at 400 KB).
        dedupe_window: Number of most recent entry uuids remembered per key
            to make re-delivered appends idempotent.
        max_commit_attempts: Manifest commit attempts under contention.
        ttl_in_seconds: Optional TTL applied to every document written. The
            state store must support TTL.
    """

    key_prefix: str = "claude-session:"
    max_chunk_bytes: int = 256 * 1024
    dedupe_window: int = 2048
    max_commit_attempts: int = 10
    ttl_in_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        if self.dedupe_window < 0:
            raise ValueError("dedupe_window must not be negative")
        if self.max_commit_attempts <= 0:
            raise ValueError("max_commit_attempts must be positive")


@dataclass(frozen=True)
class _Manifest:
    """Immutable view of a manifest document."""

    chunks: Tuple[str, ...] = ()
    recent_uuids: Tuple[str, ...] = ()
    subkeys: Tuple[str, ...] = ()
    mtime: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "_Manifest":
        return cls(
            chunks=tuple(str(c) for c in data.get("chunks", ())),
            recent_uuids=tuple(str(u) for u in data.get("recent_uuids", ())),
            subkeys=tuple(str(s) for s in data.get("subkeys", ())),
            mtime=int(data.get("mtime", 0) or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": _MANIFEST_VERSION,
            "chunks": list(self.chunks),
            "recent_uuids": list(self.recent_uuids),
            "subkeys": list(self.subkeys),
            "mtime": self.mtime,
        }


def _session_key_string(key: Mapping[str, Any]) -> str:
    """Render a ``SessionKey`` as ``project/session[/subpath]``.

    ``||`` is reserved by Dapr as the app-id key separator, so it is escaped.
    """
    parts = [str(key["project_key"]), str(key["session_id"])]
    subpath = key.get("subpath")
    if subpath:
        parts.append(str(subpath))
    return "/".join(parts).replace("||", "__")


def _entry_uuid(entry: Mapping[str, Any]) -> Optional[str]:
    value = entry.get("uuid")
    return value if isinstance(value, str) and value else None


def _chunk_entries(
    entries: Sequence[Mapping[str, Any]], max_bytes: int
) -> List[List[Mapping[str, Any]]]:
    """Split entries into ordered slices of at most ``max_bytes`` each."""
    chunks: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    size = 0
    for entry in entries:
        entry_size = len(json.dumps(entry, default=str).encode("utf-8"))
        if current and size + entry_size > max_bytes:
            chunks.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry_size
    if current:
        chunks.append(current)
    return chunks


def _now_ms() -> int:
    return int(time.time() * 1000)


class DaprSessionStore:
    """
    Claude Agent SDK ``SessionStore`` backed by a Dapr state store.

    Args:
        state: The ``StateStoreService`` to persist into. Reusing the agent's
            workflow state store (``AgentStateConfig.store``) is fine; keys
            are namespaced by ``config.key_prefix``.
        config: Optional tuning; defaults to ``DaprSessionStoreConfig()``.

    ``list_sessions`` and ``list_session_summaries`` are not implemented:
    maintaining a per-project index would serialize every append in a
    project behind one document. They are only needed for
    ``continue_conversation`` without an explicit session id.
    """

    def __init__(
        self,
        state: StateStoreService,
        *,
        config: Optional[DaprSessionStoreConfig] = None,
    ) -> None:
        self._state = state
        self._config = config or DaprSessionStoreConfig()

    @property
    def config(self) -> DaprSessionStoreConfig:
        """The store's tuning configuration."""
        return self._config

    # ------------------------------------------------------------------
    # SessionStore protocol
    # ------------------------------------------------------------------

    async def append(self, key: Mapping[str, Any], entries: List[Any]) -> None:
        """Append transcript entries in order, ignoring already-seen uuids."""
        if not entries:
            return
        manifest_key = self._manifest_key(key)
        await asyncio.to_thread(self._append_sync, manifest_key, list(entries))
        subpath = key.get("subpath")
        if subpath:
            main_key = self._manifest_key(
                {"project_key": key["project_key"], "session_id": key["session_id"]}
            )
            await asyncio.to_thread(self._register_subkey_sync, main_key, subpath)

    async def load(self, key: Mapping[str, Any]) -> Optional[List[Any]]:
        """Return all entries in append order, or ``None`` if never written."""
        return await asyncio.to_thread(self._load_sync, self._manifest_key(key))

    async def delete(self, key: Mapping[str, Any]) -> None:
        """Delete a transcript; a main-transcript delete cascades to subkeys."""
        await asyncio.to_thread(self._delete_sync, key)

    async def list_subkeys(self, key: Mapping[str, Any]) -> List[str]:
        """List subpaths (e.g. subagent transcripts) recorded for a session."""
        manifest, _ = await asyncio.to_thread(
            self._read_manifest, self._manifest_key(key)
        )
        return list(manifest.subkeys) if manifest else []

    # ------------------------------------------------------------------
    # Synchronous implementation (runs in a worker thread)
    # ------------------------------------------------------------------

    def _manifest_key(self, key: Mapping[str, Any]) -> str:
        return f"{self._config.key_prefix}{_session_key_string(key)}"

    @staticmethod
    def _chunk_key(manifest_key: str, chunk_id: str) -> str:
        return f"{manifest_key}#{chunk_id}"

    def _read_manifest(
        self, manifest_key: str
    ) -> Tuple[Optional[_Manifest], Optional[str]]:
        data, etag = self._state.load_with_etag(
            key=manifest_key, default=None, state_metadata=_STATE_METADATA
        )
        if not etag and not data:
            return None, None
        return _Manifest.from_dict(data if isinstance(data, Mapping) else {}), etag

    def _write(self, key: str, value: Dict[str, Any], etag: Optional[str]) -> None:
        self._state.save(
            key=key,
            value=value,
            etag=etag,
            state_metadata=_STATE_METADATA,
            state_options=_SAVE_OPTIONS if etag is not None else None,
            ttl_in_seconds=self._config.ttl_in_seconds,
        )

    def _commit_manifest(
        self, manifest_key: str, manifest: _Manifest, etag: Optional[str]
    ) -> None:
        if etag is None:
            # First write: first-write concurrency without an ETag only
            # succeeds when the key does not exist yet.
            self._state.save(
                key=manifest_key,
                value=manifest.to_dict(),
                state_metadata=_STATE_METADATA,
                state_options=_SAVE_OPTIONS,
                ttl_in_seconds=self._config.ttl_in_seconds,
            )
            return
        self._write(manifest_key, manifest.to_dict(), etag)

    def _new_entries(
        self, manifest: Optional[_Manifest], entries: Sequence[Any]
    ) -> List[Any]:
        seen = set(manifest.recent_uuids) if manifest else set()
        fresh: List[Any] = []
        for entry in entries:
            entry_id = _entry_uuid(entry) if isinstance(entry, Mapping) else None
            if entry_id is not None:
                if entry_id in seen:
                    continue
                seen.add(entry_id)
            fresh.append(entry)
        return fresh

    def _write_chunks(self, manifest_key: str, entries: Sequence[Any]) -> List[str]:
        chunk_ids: List[str] = []
        for chunk in _chunk_entries(entries, self._config.max_chunk_bytes):
            chunk_id = uuid.uuid4().hex
            self._write(
                self._chunk_key(manifest_key, chunk_id),
                {"entries": list(chunk)},
                None,
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    def _next_manifest(
        self,
        manifest: Optional[_Manifest],
        chunk_ids: Sequence[str],
        entries: Sequence[Any],
    ) -> _Manifest:
        base = manifest or _Manifest()
        new_uuids = [
            u
            for u in (_entry_uuid(e) for e in entries if isinstance(e, Mapping))
            if u is not None
        ]
        window = self._config.dedupe_window
        recent = (*base.recent_uuids, *new_uuids)
        return _Manifest(
            chunks=(*base.chunks, *chunk_ids),
            recent_uuids=tuple(recent[-window:]) if window else (),
            subkeys=base.subkeys,
            mtime=max(_now_ms(), base.mtime + 1),
        )

    def _append_sync(self, manifest_key: str, entries: List[Any]) -> None:
        attempts = self._config.max_commit_attempts
        for attempt in range(1, attempts + 1):
            manifest, etag = self._read_manifest(manifest_key)
            fresh = self._new_entries(manifest, entries)
            if not fresh:
                return
            chunk_ids = self._write_chunks(manifest_key, fresh)
            try:
                self._commit_manifest(
                    manifest_key, self._next_manifest(manifest, chunk_ids, fresh), etag
                )
                return
            except StateStoreError as exc:
                self._delete_chunks(manifest_key, chunk_ids)
                if attempt == attempts:
                    raise
                logger.debug(
                    "Session manifest commit conflict for %s (attempt %d/%d): %s",
                    manifest_key,
                    attempt,
                    attempts,
                    exc,
                )
                _backoff(attempt)

    def _register_subkey_sync(self, main_key: str, subpath: str) -> None:
        attempts = self._config.max_commit_attempts
        for attempt in range(1, attempts + 1):
            manifest, etag = self._read_manifest(main_key)
            base = manifest or _Manifest()
            if subpath in base.subkeys:
                return
            updated = _Manifest(
                chunks=base.chunks,
                recent_uuids=base.recent_uuids,
                subkeys=(*base.subkeys, subpath),
                mtime=max(_now_ms(), base.mtime + 1),
            )
            try:
                self._commit_manifest(main_key, updated, etag)
                return
            except StateStoreError:
                if attempt == attempts:
                    raise
                _backoff(attempt)

    def _load_sync(self, manifest_key: str) -> Optional[List[Any]]:
        manifest, _ = self._read_manifest(manifest_key)
        if manifest is None:
            return None
        if not manifest.chunks:
            return []
        chunk_keys = [self._chunk_key(manifest_key, c) for c in manifest.chunks]
        docs = self._state.load_many(chunk_keys, state_metadata=_STATE_METADATA)
        entries: List[Any] = []
        for chunk_key in chunk_keys:
            doc = docs.get(chunk_key)
            if not isinstance(doc, Mapping):
                raise StateStoreError(
                    f"Session transcript chunk '{chunk_key}' is missing; "
                    "refusing to return a transcript with a gap."
                )
            entries.extend(doc.get("entries", ()))
        return entries

    def _delete_chunks(self, manifest_key: str, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            try:
                self._state.delete(key=self._chunk_key(manifest_key, chunk_id))
            except StateStoreError:
                logger.warning(
                    "Failed to delete session chunk %s for %s",
                    chunk_id,
                    manifest_key,
                    exc_info=True,
                )

    def _delete_one(self, manifest_key: str) -> Optional[_Manifest]:
        manifest, _ = self._read_manifest(manifest_key)
        if manifest is None:
            return None
        self._state.delete(key=manifest_key)
        self._delete_chunks(manifest_key, manifest.chunks)
        return manifest

    def _delete_sync(self, key: Mapping[str, Any]) -> None:
        manifest = self._delete_one(self._manifest_key(key))
        if key.get("subpath") or manifest is None:
            return
        for subpath in manifest.subkeys:
            self._delete_one(self._manifest_key({**key, "subpath": subpath}))


def _backoff(attempt: int) -> None:
    time.sleep(min(0.05 * attempt, 0.5) * (1 + random.uniform(0, 0.25)))
