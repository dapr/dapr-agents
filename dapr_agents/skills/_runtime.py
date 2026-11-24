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

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dapr_agents.skills._constants import (
    LOAD_SKILL_TOOL_NAME,
    READ_SKILL_RESOURCE_TOOL_NAME,
)
from dapr_agents.skills.errors import (
    SkillAccessError,
    SkillCatalogError,
    SkillError,
    SkillMalformedError,
    SkillNotFoundError,
    SkillSizeLimitError,
    SkillUnsupportedMediaTypeError,
    SkillUnsafePathError,
)
from dapr_agents.skills.models import (
    MAX_RESOURCE_BYTES,
    MAX_RUN_RESOURCE_BYTES,
    Skill,
    SkillMetadata,
    SkillResource,
    _validate_skill_path,
)
from dapr_agents.skills.source import SkillSource

# These names are part of the persisted tool-history protocol. They are
# deliberately explicit rather than generic aliases: history restoration must
# distinguish an activation result from a resource result before accepting a
# cached value. The leading underscore keeps these implementation details
# private to the skills package.
_ModelT = TypeVar("_ModelT", bound=BaseModel)

_TEXT_MEDIA_TYPES = {
    "application/json",
    "application/javascript",
    "application/ld+json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
}


def _skill_metadata(skill: Skill) -> SkillMetadata:
    """Build catalog metadata for one direct skill."""
    return SkillMetadata(
        name=skill.name,
        description=skill.description,
        license=skill.license,
        compatibility=skill.compatibility,
        metadata=dict(skill.metadata),
        extra_frontmatter=dict(skill.extra_frontmatter),
    )


class LoadSkillResult(BaseModel):
    """Structured result returned by ``load_skill``."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    instructions: StrictStr
    available_resources: list[StrictStr] = Field(default_factory=list)


class ReadSkillResourceResult(BaseModel):
    """Structured UTF-8 result returned by ``read_skill_resource``."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    path: StrictStr
    media_type: StrictStr
    encoding: StrictStr = "utf-8"
    content: StrictStr


@dataclass(frozen=True)
class _SkillCatalogEntry:
    """Internal catalog entry retaining provenance for skill-source routing."""

    metadata: SkillMetadata
    # Internal source-routing identity; never expose it to the model. The
    # model-facing identity remains the ordinary skill name.
    source_id: str

    def to_dict(self) -> Dict[str, Any]:
        result = self.metadata.model_dump(mode="json")
        # Provenance is consumed only by the runtime adapter and is never
        # rendered in the model-facing Available Skills section.
        result["_source_id"] = self.source_id
        return result


@dataclass
class _RunCache:
    """In-memory skill state isolated to one workflow execution.

    The lock protects cache and in-flight maps across concurrent tool calls.
    In-flight futures let duplicate requests await one source operation,
    while durable history can rehydrate successful results after worker restart.
    """

    loaded: Dict[str, LoadSkillResult] = field(default_factory=dict)
    resources: Dict[tuple[str, str], ReadSkillResourceResult] = field(
        default_factory=dict
    )
    resource_bytes: int = 0
    history_rehydrated: bool = False

    lock: threading.Lock = field(default_factory=threading.Lock)
    load_inflight: Dict[str, Future[LoadSkillResult]] = field(default_factory=dict)
    resource_inflight: Dict[tuple[str, str], Future[ReadSkillResourceResult]] = field(
        default_factory=dict
    )


class _DirectSkillSource:
    """Adapt one direct in-memory ``Skill`` to the ``SkillSource`` protocol."""

    def __init__(self, skill: Skill) -> None:
        self._skill = skill
        self.source_id = f"direct:{skill.name}"

    async def list_metadata(self) -> Sequence[SkillMetadata]:
        return [_skill_metadata(self._skill)]

    async def load_skill(self, name: str) -> Skill:
        if name != self._skill.name:
            raise SkillNotFoundError(skill_name=name)
        return self._skill

    async def read_resource(self, name: str, path: str) -> SkillResource:
        if name != self._skill.name or path not in self._skill.resources:
            raise SkillNotFoundError(skill_name=name, path=path)
        return self._skill.resources[path]


class SkillRuntimeAdapter:
    """Adapt configured skills and sources to the run-scoped skill runtime.

    The adapter combines direct ``Skill`` values, whose metadata is materialized
    eagerly, with lazy ``SkillSource`` values. It adds stable catalog provenance
    and owns per-workflow activation, resource, and history caches. Keeping these
    concerns here ensures resource reads cross the ``load_skill`` activation
    boundary first.
    """

    def __init__(self, skills: Sequence[Skill | SkillSource]) -> None:
        # Keep sources and their stable provenance IDs in matching order.
        self._sources: list[SkillSource] = []
        self._source_ids: list[str] = []

        # Deduplicate repeated object references without treating source order as
        # precedence; distinct ambiguous source identities fail during discovery.
        seen_source_objects: set[int] = set()

        # Materialize direct Skill metadata immediately. Defer all SkillSource I/O
        # until run-start discovery.
        self._direct_catalog: list[_SkillCatalogEntry] = []
        for value in skills:
            source: SkillSource
            if isinstance(value, Skill):
                source = _DirectSkillSource(value)
                self._direct_catalog.append(
                    _SkillCatalogEntry(_skill_metadata(value), source.source_id)
                )
            elif isinstance(value, SkillSource):
                if id(value) in seen_source_objects:
                    continue
                seen_source_objects.add(id(value))
                source = value
            else:
                raise TypeError(
                    "skills must contain only Skill or SkillSource values, "
                    f"got {type(value).__name__}"
                )
            self._sources.append(source)
            self._source_ids.append(self._source_identity(source))

        # Keep activation, resource, budget, and in-flight state isolated per
        # workflow execution and protect the run registry across worker threads.
        self._runs: Dict[str, _RunCache] = {}
        self._runs_lock = threading.Lock()

    @staticmethod
    def _source_identity(source: SkillSource) -> str:
        """Return stable internal provenance for one configured source."""
        explicit = getattr(source, "source_id", None)
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip():
                raise TypeError("SkillSource.source_id must be a non-empty string")
            return explicit
        # A qualified class name is stable across worker instances. Do not
        # append a positional index: source order can change between runs, so
        # duplicate source classes must provide explicit provenance instead.
        source_type = type(source)
        return f"{source_type.__module__}:{source_type.__qualname__}"

    def _run_cache(self, instance_id: str) -> _RunCache:
        """Return the synchronized cache for one workflow execution."""
        with self._runs_lock:
            return self._runs.setdefault(instance_id, _RunCache())

    @staticmethod
    def _fail_inflight(
        inflight: Dict[Any, Future[Any]],
        key: Any,
        pending: Future[Any],
        error: BaseException,
    ) -> None:
        """Fail a pending call; the caller must hold the run-cache lock."""
        inflight.pop(key, None)
        if not pending.done():
            pending.set_exception(error)
            # Mark the exception observed so a failed owner does not emit an
            # unhandled-future warning before waiters receive it.
            pending.exception()

    @staticmethod
    def _complete_inflight(
        inflight: Dict[Any, Future[Any]], key: Any, pending: Future[Any], result: Any
    ) -> None:
        """Complete a pending call; the caller must hold the run-cache lock."""
        inflight.pop(key, None)
        if not pending.done():
            pending.set_result(result)

    async def discover_catalog(self) -> list[dict[str, Any]]:
        """Merge eager direct metadata with discovered source metadata."""
        # Persisted catalog entries route later calls by source ID, so duplicate
        # IDs would make activation and resource reads ambiguous.
        if len(set(self._source_ids)) != len(self._source_ids):
            raise SkillCatalogError()

        catalog: list[_SkillCatalogEntry] = []
        names: set[str] = set()
        for entry in self._direct_catalog:
            if entry.metadata.name in names:
                raise SkillCatalogError(skill_name=entry.metadata.name)
            names.add(entry.metadata.name)
            catalog.append(entry)

        for source, source_id in zip(self._sources, self._source_ids):
            # Direct entries were materialized above; only external sources
            # may perform discovery I/O here.
            if isinstance(source, _DirectSkillSource):
                continue
            catalog.extend(
                await self._discover_source_catalog(source, source_id, names)
            )

        try:
            return [entry.to_dict() for entry in catalog]
        except Exception as exc:  # noqa: BLE001 - safe public category
            raise SkillCatalogError() from exc

    async def _discover_source_catalog(
        self,
        source: SkillSource,
        source_id: str,
        names: set[str],
    ) -> list[_SkillCatalogEntry]:
        """Discover and validate catalog entries from one configured source."""
        try:
            metadata_items = await source.list_metadata()
        except SkillError:
            raise
        except Exception as exc:  # noqa: BLE001 - redact source failures
            # A partial catalog could omit a duplicate or expose an incomplete
            # capability set, so discovery is intentionally all-or-nothing.
            raise SkillAccessError() from exc

        entries: list[_SkillCatalogEntry] = []
        try:
            for raw_metadata in metadata_items:
                metadata = SkillMetadata.model_validate(raw_metadata)
                if metadata.name in names:
                    raise SkillCatalogError(skill_name=metadata.name)
                names.add(metadata.name)
                entries.append(_SkillCatalogEntry(metadata, source_id))
        except SkillError:
            raise
        except Exception as exc:  # noqa: BLE001 - safe public category
            raise SkillCatalogError() from exc
        return entries

    async def load(
        self,
        instance_id: str,
        name: str,
        catalog: Sequence[Mapping[str, Any]],
        history: Sequence[Any] | None = None,
    ) -> LoadSkillResult:
        """Activate one configured skill, coalescing concurrent requests."""
        run_cache = self._run_cache(instance_id)
        with run_cache.lock:
            # Rehydrate before checking the cache so worker restart can reuse
            # previously validated activation results.
            self._restore_history(run_cache, history)
            cached = run_cache.loaded.get(name)
            if cached is not None:
                return cached

            # Register one in-flight load under the lock; later callers wait
            # for its result instead of starting another source call.
            pending = run_cache.load_inflight.get(name)
            if pending is None:
                pending = Future()
                run_cache.load_inflight[name] = pending
                owner = True
            else:
                owner = False

        if not owner:
            # Concurrent requests for the same skill share the owner's source
            # call instead of duplicating I/O or cache population.
            return await asyncio.wrap_future(pending)

        try:
            result = await self._fetch_skill(name, catalog)
        except BaseException as exc:
            with run_cache.lock:
                self._fail_inflight(run_cache.load_inflight, name, pending, exc)
            raise

        with run_cache.lock:
            # Publish the result and release all concurrent waiters atomically.
            run_cache.loaded[name] = result
            self._complete_inflight(run_cache.load_inflight, name, pending, result)
        return result

    async def _fetch_skill(
        self, name: str, catalog: Sequence[Mapping[str, Any]]
    ) -> LoadSkillResult:
        """Load and validate one skill from its catalog-selected source."""
        entry = self._catalog_entry(name, catalog)
        source = self._source_for(entry)
        raw_skill = await self._call_source(
            lambda: source.load_skill(name), skill_name=name
        )
        skill = self._validate_source_model(raw_skill, Skill, skill_name=name)
        if skill.name != name:
            raise SkillMalformedError(skill_name=name)
        return LoadSkillResult(
            name=skill.name,
            instructions=skill.instructions,
            available_resources=list(skill.resources),
        )

    async def read_resource(
        self,
        instance_id: str,
        name: str,
        path: str,
        catalog: Sequence[Mapping[str, Any]],
        history: Sequence[Any] | None = None,
    ) -> ReadSkillResourceResult:
        """Read one UTF-8 resource from an activated skill, coalescing concurrent requests."""
        try:
            _validate_skill_path(path)
        except ValueError as exc:
            raise SkillUnsafePathError(skill_name=name, path=path) from exc

        run_cache = self._run_cache(instance_id)
        with run_cache.lock:
            # Rehydrate before checking the cache so worker restart can reuse
            # previously validated resource results and budget usage.
            self._restore_history(run_cache, history)
            cached = run_cache.resources.get((name, path))
            if cached is not None:
                return cached
            # Resource reads require activation in this workflow execution;
            # source access alone must not bypass the load_skill boundary.
            loaded = run_cache.loaded.get(name)
            if loaded is None:
                raise SkillAccessError(skill_name=name, path=path)
            if loaded.available_resources and path not in loaded.available_resources:
                raise SkillNotFoundError(skill_name=name, path=path)

            # Register one in-flight read under the lock; later callers wait
            # for its result instead of starting another source read.
            pending = run_cache.resource_inflight.get((name, path))
            if pending is None:
                pending = Future()
                run_cache.resource_inflight[(name, path)] = pending
                owner = True
            else:
                owner = False

        if not owner:
            # Wait for the owner instead of issuing a duplicate source read.
            return await asyncio.wrap_future(pending)

        try:
            result, size = await self._fetch_resource(name, path, catalog)
        except BaseException as exc:
            with run_cache.lock:
                self._fail_inflight(
                    run_cache.resource_inflight, (name, path), pending, exc
                )
            raise

        with run_cache.lock:
            cached = run_cache.resources.get((name, path))
            if cached is not None:
                # Another waiter may have populated the cache while this call
                # was reading; charge the byte budget only on first insertion.
                result = cached
            elif run_cache.resource_bytes + size > MAX_RUN_RESOURCE_BYTES:
                error = SkillSizeLimitError(skill_name=name, path=path)
                self._fail_inflight(
                    run_cache.resource_inflight, (name, path), pending, error
                )
                raise error
            else:
                run_cache.resources[(name, path)] = result
                run_cache.resource_bytes += size
            # Publish the result and release all concurrent waiters atomically.
            self._complete_inflight(
                run_cache.resource_inflight, (name, path), pending, result
            )
        return result

    async def _fetch_resource(
        self,
        name: str,
        path: str,
        catalog: Sequence[Mapping[str, Any]],
    ) -> tuple[ReadSkillResourceResult, int]:
        """Read, validate, decode, and size one resource."""
        entry = self._catalog_entry(name, catalog)
        source = self._source_for(entry)
        raw_resource = await self._call_source(
            lambda: source.read_resource(name, path),
            skill_name=name,
            path=path,
        )
        resource = self._validate_source_model(
            raw_resource, SkillResource, skill_name=name, path=path
        )
        if resource.path != path:
            raise SkillMalformedError(skill_name=name, path=path)
        result = self._to_text_result(name, resource)
        return result, len(result.content.encode("utf-8"))

    @staticmethod
    async def _call_source(
        operation: Callable[[], Awaitable[_ModelT]],
        *,
        skill_name: str,
        path: str | None = None,
    ) -> _ModelT:
        """Run source I/O and map source failures to safe public errors."""
        try:
            return await operation()
        except SkillError:
            raise
        except Exception as exc:  # noqa: BLE001 - redact source failures
            raise SkillAccessError(skill_name=skill_name, path=path) from exc

    @staticmethod
    def _validate_source_model(
        value: Any,
        model_type: type[_ModelT],
        *,
        skill_name: str,
        path: str | None = None,
    ) -> _ModelT:
        """Validate source output without exposing source details."""
        try:
            return model_type.model_validate(value)
        except Exception as exc:  # noqa: BLE001 - safe public category
            raise SkillMalformedError(skill_name=skill_name, path=path) from exc

    def clear(self, instance_id: str) -> None:
        """Discard run-scoped runtime state for a completed workflow execution."""
        with self._runs_lock:
            self._runs.pop(instance_id, None)

    @staticmethod
    def _catalog_entry(
        name: str, catalog: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        for entry in catalog:
            if entry.get("name") == name:
                return entry
        raise SkillNotFoundError(skill_name=name)

    def _source_for(self, entry: Mapping[str, Any]) -> SkillSource:
        """Resolve the configured skill source from stable catalog provenance."""
        source_id = entry.get("_source_id")
        if not isinstance(source_id, str):
            raise SkillCatalogError(skill_name=str(entry.get("name", "unknown")))
        try:
            return self._sources[self._source_ids.index(source_id)]
        except ValueError as exc:
            raise SkillCatalogError(
                skill_name=str(entry.get("name", "unknown"))
            ) from exc

    @staticmethod
    def _to_text_result(name: str, resource: SkillResource) -> ReadSkillResourceResult:
        media_type = resource.media_type.split(";", 1)[0].strip().lower()
        if not (media_type.startswith("text/") or media_type in _TEXT_MEDIA_TYPES):
            raise SkillUnsupportedMediaTypeError(skill_name=name, path=resource.path)
        if isinstance(resource.content, bytes):
            try:
                content = resource.content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SkillUnsupportedMediaTypeError(
                    skill_name=name, path=resource.path
                ) from exc
        else:
            content = resource.content
        return ReadSkillResourceResult(
            name=name,
            path=resource.path,
            media_type=resource.media_type,
            content=content,
        )

    @staticmethod
    def _restore_history(run_cache: _RunCache, history: Sequence[Any] | None) -> None:
        """Rehydrate skill activation and resource state from conversation history.

        The conversation layer supplies the reconstructed history. Restore
        successful ``load_skill`` results in a first pass so a resource result
        cannot bypass activation merely because messages were recorded out of
        order. The second pass can then safely validate resource provenance and
        byte budgets against the restored activation state.
        """
        if run_cache.history_rehydrated or not history:
            return

        for message in history:
            if SkillRuntimeAdapter._message_name(message) == LOAD_SKILL_TOOL_NAME:
                SkillRuntimeAdapter._restore_history_message(run_cache, message)

        for message in history:
            if (
                SkillRuntimeAdapter._message_name(message)
                == READ_SKILL_RESOURCE_TOOL_NAME
            ):
                SkillRuntimeAdapter._restore_history_message(run_cache, message)
        run_cache.history_rehydrated = True

    @staticmethod
    def _message_name(message: Any) -> Any:
        return (
            message.get("name")
            if isinstance(message, dict)
            else getattr(message, "name", None)
        )

    @staticmethod
    def _restore_history_message(run_cache: _RunCache, message: Any) -> None:
        """Restore one valid skill tool result from conversation history."""
        name = (
            message.get("name")
            if isinstance(message, dict)
            else getattr(message, "name", None)
        )
        # Ignore unrelated tool messages; only skill results can rebuild this cache.
        if name not in {
            LOAD_SKILL_TOOL_NAME,
            READ_SKILL_RESOURCE_TOOL_NAME,
        }:
            return
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        # Only serialized structured results are eligible for history hydration.
        if not isinstance(content, str):
            return

        try:
            payload = json.loads(content)

            if name == LOAD_SKILL_TOOL_NAME:
                load_result = LoadSkillResult.model_validate(payload)
                run_cache.loaded.setdefault(load_result.name, load_result)
            else:
                resource_result = ReadSkillResourceResult.model_validate(payload)
                # Ignore resource results without valid activation; read_resource
                # will fail closed.
                loaded = run_cache.loaded.get(resource_result.name)
                if loaded is None:
                    return

                if (
                    loaded.available_resources
                    and resource_result.path not in loaded.available_resources
                ):
                    return

                # Recheck historical results against current resource limits.
                size = len(resource_result.content.encode("utf-8"))
                if size > MAX_RESOURCE_BYTES:
                    raise SkillSizeLimitError(
                        skill_name=resource_result.name, path=resource_result.path
                    )

                key = (resource_result.name, resource_result.path)
                # Count each resource once when rebuilding the run budget.
                if key not in run_cache.resources:
                    if run_cache.resource_bytes + size > MAX_RUN_RESOURCE_BYTES:
                        raise SkillSizeLimitError(
                            skill_name=resource_result.name, path=resource_result.path
                        )
                    run_cache.resources[key] = resource_result
                    run_cache.resource_bytes += size
        except (TypeError, ValueError):
            # Legacy or malformed history is not a cache hit; the normal source
            # path will validate and reload it safely.
            return
