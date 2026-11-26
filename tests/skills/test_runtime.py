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

import asyncio
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from dapr_agents.skills import (
    MAX_RESOURCE_BYTES,
    MAX_RUN_RESOURCE_BYTES,
    Skill,
    SkillAccessError,
    SkillCatalogError,
    SkillMalformedError,
    SkillMetadata,
    SkillNotFoundError,
    SkillUnsafePathError,
    SkillResource,
    SkillSizeLimitError,
    SkillUnsupportedMediaTypeError,
)
from dapr_agents.skills._runtime import (
    LoadSkillResult,
    ReadSkillResourceResult,
    SkillRuntimeAdapter,
)


class InMemorySource:
    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.load_calls = 0
        self.read_calls = 0

    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        expected = f"memory:{self.skill.name}"
        if snapshot_id not in (None, expected):
            raise SkillAccessError(skill_name=self.skill.name)
        return expected

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        return [
            SkillMetadata(
                name=self.skill.name,
                description=self.skill.description,
                license=self.skill.license,
                compatibility=self.skill.compatibility,
                metadata=dict(self.skill.metadata),
                extra_frontmatter=dict(self.skill.extra_frontmatter),
            )
        ]

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        self.load_calls += 1
        assert name == self.skill.name
        return self.skill

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        self.read_calls += 1
        assert name == self.skill.name
        return self.skill.resources[path]


class LazyResourceSource(InMemorySource):
    def __init__(self, skill: Skill) -> None:
        super().__init__(skill.model_copy(update={"resources": {}}))
        self.resource = next(iter(skill.resources.values()))

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        self.read_calls += 1
        assert name == self.skill.name
        assert path == self.resource.path
        return self.resource


class FailingSource(InMemorySource):
    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise RuntimeError("credential=secret-token locator=private-bucket")


class NamedSource(InMemorySource):
    def __init__(self, skill: Skill, source_id: str) -> None:
        super().__init__(skill)
        self.source_id = source_id


class RecordingSource(InMemorySource):
    def __init__(self, skill: Skill) -> None:
        super().__init__(skill)
        self.read_paths: list[str] = []

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        self.read_paths.append(path)
        return await super().read_resource(name, path, snapshot_id=snapshot_id)


class MalformedMetadataSource:
    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        return "malformed"

    async def list_metadata(self, *, snapshot_id: str | None = None) -> None:
        return None

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise AssertionError(name)

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise AssertionError((name, path))


class MalformedLoadSource(InMemorySource):
    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Any:
        return {"name": name}


class MismatchedLoadSource(InMemorySource):
    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        return self.skill.model_copy(update={"name": "different-skill"})


class SkillErrorLoadSource(InMemorySource):
    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise SkillAccessError(skill_name=name)


class SkillErrorResourceSource(InMemorySource):
    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise SkillAccessError(skill_name=name, path=path)


class FailingResourceSource(InMemorySource):
    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise RuntimeError("credential=secret-token")


class MalformedResourceSource(InMemorySource):
    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> Any:
        return {"path": path}


class MismatchedResourceSource(InMemorySource):
    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        return SkillResource(path="other.md", content="wrong")


class BlockingSource(InMemorySource):
    def __init__(self, skill: Skill) -> None:
        super().__init__(skill)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        self.load_calls += 1
        self.started.set()
        await self.release.wait()
        return self.skill


class BlockingResourceSource(InMemorySource):
    def __init__(self, skill: Skill, expected_reads: int = 1) -> None:
        super().__init__(skill)
        self.expected_reads = expected_reads
        self.started_reads = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        self.read_calls += 1
        self.started_reads += 1
        if self.started_reads == self.expected_reads:
            self.started.set()
        await self.release.wait()
        return self.skill.resources[path]


class MultiSkillSource:
    def __init__(self, skills: Sequence[Skill]) -> None:
        self.skills = {skill.name: skill for skill in skills}
        self.load_calls = 0

    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        expected = "memory:" + ",".join(sorted(self.skills))
        if snapshot_id not in (None, expected):
            raise SkillAccessError()
        return expected

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        return [
            SkillMetadata(name=skill.name, description=skill.description)
            for skill in self.skills.values()
        ]

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        self.load_calls += 1
        return self.skills[name]

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        return self.skills[name].resources[path]


class SkillErrorMetadataSource:
    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        return "error"

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        raise SkillAccessError()

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise AssertionError(name)

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise AssertionError((name, path))


class FailingMetadataSource:
    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        return "failure"

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        raise RuntimeError("credential=secret-token locator=private-bucket")

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise AssertionError(name)

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise AssertionError((name, path))


@pytest.fixture
def skill() -> Skill:
    return Skill(
        name="incident-investigation",
        description="Investigate incidents.",
        instructions="Follow the checklist.",
        resources={
            "references/checklist.md": SkillResource(
                path="references/checklist.md",
                content="# Checklist",
                media_type="text/markdown",
            )
        },
    )


@pytest.mark.asyncio
async def test_discover_catalog_merges_direct_and_source_metadata(skill: Skill) -> None:
    source_skill = skill.model_copy(update={"name": "release-notes"})
    source = InMemorySource(source_skill)
    adapter = SkillRuntimeAdapter([skill, source])

    catalog = await adapter.discover_catalog()

    assert [entry["name"] for entry in catalog] == [
        "incident-investigation",
        "release-notes",
    ]
    assert all("source_id" not in entry for entry in catalog)


@pytest.mark.asyncio
async def test_mixed_catalog_routes_direct_and_source_activation(skill: Skill) -> None:
    source_skill = skill.model_copy(update={"name": "release-notes"})
    source = InMemorySource(source_skill)
    adapter = SkillRuntimeAdapter([skill, source])
    catalog = await adapter.discover_catalog()

    direct = await adapter.load("run-1", skill.name, catalog)
    sourced = await adapter.load("run-1", source_skill.name, catalog)

    assert direct.instructions == skill.instructions
    assert sourced.instructions == source_skill.instructions
    assert source.load_calls == 1


@pytest.mark.asyncio
async def test_mixed_catalog_duplicate_names_fail_closed(skill: Skill) -> None:
    with pytest.raises(SkillCatalogError):
        await SkillRuntimeAdapter([skill, InMemorySource(skill)]).discover_catalog()


@pytest.mark.asyncio
async def test_source_can_return_multiple_skill_metadata(skill: Skill) -> None:
    first = skill.model_copy(update={"name": "first-skill"})
    second = skill.model_copy(update={"name": "second-skill"})
    adapter = SkillRuntimeAdapter([MultiSkillSource([first, second])])

    catalog = await adapter.discover_catalog()

    assert [entry["name"] for entry in catalog] == ["first-skill", "second-skill"]


@pytest.mark.asyncio
async def test_direct_metadata_is_materialized_before_catalog_discovery(
    skill: Skill,
) -> None:
    """Direct metadata is snapshotted before provider discovery runs."""
    adapter = SkillRuntimeAdapter([skill])
    skill.description = "Changed after configuration."

    catalog = await adapter.discover_catalog()

    assert catalog[0]["description"] == "Investigate incidents."


@pytest.mark.asyncio
async def test_duplicate_skill_names_fail_closed(skill: Skill) -> None:
    duplicate = skill.model_copy(update={"description": "different"})

    with pytest.raises(SkillCatalogError):
        await SkillRuntimeAdapter([skill, duplicate]).discover_catalog()


@pytest.mark.asyncio
async def test_load_is_cached_per_workflow_instance(skill: Skill) -> None:
    source = InMemorySource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()

    first = await adapter.load("run-1", skill.name, catalog)
    second = await adapter.load("run-1", skill.name, catalog)

    assert isinstance(first, LoadSkillResult)
    assert first == second
    assert first.available_resources == ["references/checklist.md"]
    assert source.load_calls == 1


@pytest.mark.asyncio
async def test_resource_read_requires_activation_and_returns_utf8_text(
    skill: Skill,
) -> None:
    adapter = SkillRuntimeAdapter([skill])
    catalog = await adapter.discover_catalog()

    with pytest.raises(SkillAccessError):
        await adapter.read_resource(
            "run-1", skill.name, "references/checklist.md", catalog
        )

    await adapter.load("run-1", skill.name, catalog)
    result = await adapter.read_resource(
        "run-1", skill.name, "references/checklist.md", catalog
    )

    assert isinstance(result, ReadSkillResourceResult)
    assert result.encoding == "utf-8"
    assert result.content == "# Checklist"


@pytest.mark.asyncio
async def test_resource_read_allows_lazy_source_without_resource_inventory(
    skill: Skill,
) -> None:
    source = LazyResourceSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()

    await adapter.load("run-1", skill.name, catalog)
    result = await adapter.read_resource(
        "run-1", skill.name, "references/checklist.md", catalog
    )

    assert result.content == "# Checklist"
    assert source.read_calls == 1


@pytest.mark.asyncio
async def test_history_resource_result_cannot_bypass_activation(skill: Skill) -> None:
    adapter = SkillRuntimeAdapter([skill])
    catalog = await adapter.discover_catalog()
    history = [
        {
            "role": "tool",
            "name": "read_skill_resource",
            "content": ReadSkillResourceResult(
                name=skill.name,
                path="references/checklist.md",
                media_type="text/markdown",
                content="# History",
            ).model_dump_json(),
        }
    ]

    with pytest.raises(SkillAccessError):
        await adapter.read_resource(
            "run-1", skill.name, "references/checklist.md", catalog, history
        )


@pytest.mark.asyncio
async def test_history_resource_results_enforce_run_budget(skill: Skill) -> None:
    resources = {
        f"reference-{index}.txt": SkillResource(
            path=f"reference-{index}.txt",
            content=b"x" * MAX_RESOURCE_BYTES,
            media_type="text/plain",
        )
        for index in range((MAX_RUN_RESOURCE_BYTES // MAX_RESOURCE_BYTES) + 1)
    }
    large_skill = Skill(
        name="history-budget-skill",
        description="History budget.",
        instructions="Use history.",
        resources=resources,
    )
    adapter = SkillRuntimeAdapter([large_skill])
    catalog = await adapter.discover_catalog()
    loaded = await adapter.load("run-1", large_skill.name, catalog)
    history = [
        {
            "role": "tool",
            "name": "load_skill",
            "content": loaded.model_dump_json(),
        },
        *[
            {
                "role": "tool",
                "name": "read_skill_resource",
                "content": ReadSkillResourceResult(
                    name=large_skill.name,
                    path=resource.path,
                    media_type=resource.media_type,
                    content="x" * MAX_RESOURCE_BYTES,
                ).model_dump_json(),
            }
            for resource in resources.values()
        ],
    ]

    restored = SkillRuntimeAdapter([large_skill])
    with pytest.raises(SkillSizeLimitError):
        await restored.load("run-1", large_skill.name, catalog, history)


@pytest.mark.asyncio
async def test_binary_resource_is_rejected_without_base64(skill: Skill) -> None:
    binary_skill = skill.model_copy(
        update={
            "resources": {
                "data.bin": SkillResource(
                    path="data.bin",
                    content=b"\x00binary",
                    media_type="application/octet-stream",
                )
            }
        }
    )
    adapter = SkillRuntimeAdapter([binary_skill])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", binary_skill.name, catalog)

    with pytest.raises(SkillUnsupportedMediaTypeError):
        await adapter.read_resource("run-1", binary_skill.name, "data.bin", catalog)


@pytest.mark.asyncio
async def test_utf8_resource_with_unknown_extension_is_model_readable(
    skill: Skill,
) -> None:
    text_skill = skill.model_copy(
        update={
            "resources": {
                "notes.data": SkillResource(
                    path="notes.data",
                    content=b"plain UTF-8 text",
                    media_type="application/octet-stream",
                )
            }
        }
    )
    adapter = SkillRuntimeAdapter([text_skill])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", text_skill.name, catalog)

    result = await adapter.read_resource(
        "run-1", text_skill.name, "notes.data", catalog
    )

    assert result.content == "plain UTF-8 text"


@pytest.mark.asyncio
async def test_history_fallback_reconstructs_results_without_source_calls(
    skill: Skill,
) -> None:
    source = InMemorySource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    loaded = await adapter.load("run-1", skill.name, catalog)
    resource = await adapter.read_resource(
        "run-1", skill.name, "references/checklist.md", catalog
    )
    history = [
        {"role": "tool", "name": "load_skill", "content": loaded.model_dump_json()},
        {
            "role": "tool",
            "name": "read_skill_resource",
            "content": resource.model_dump_json(),
        },
    ]

    source.load_calls = 0
    source.read_calls = 0
    restored = SkillRuntimeAdapter([source])
    restored_loaded = await restored.load("run-1", skill.name, catalog, history)
    restored_resource = await restored.read_resource(
        "run-1", skill.name, "references/checklist.md", catalog, history
    )

    assert restored_loaded == loaded
    assert restored_resource == resource
    assert source.load_calls == 0
    assert source.read_calls == 0


@pytest.mark.asyncio
async def test_cache_and_budget_are_isolated_by_workflow_instance() -> None:
    resource = SkillResource(
        path="reference.txt",
        content=b"x" * MAX_RESOURCE_BYTES,
        media_type="text/plain",
    )
    skill = Skill(
        name="large-skill",
        description="Large resources.",
        instructions="Use resources.",
        resources={"reference.txt": resource},
    )
    adapter = SkillRuntimeAdapter([skill])
    catalog = await adapter.discover_catalog()

    await adapter.load("run-1", skill.name, catalog)
    await adapter.load("run-2", skill.name, catalog)
    first = await adapter.read_resource("run-1", skill.name, resource.path, catalog)
    second = await adapter.read_resource("run-2", skill.name, resource.path, catalog)

    assert first.content == second.content == "x" * MAX_RESOURCE_BYTES


@pytest.mark.asyncio
async def test_source_failure_is_redacted(skill: Skill) -> None:
    adapter = SkillRuntimeAdapter([FailingSource(skill)])
    catalog = await adapter.discover_catalog()

    with pytest.raises(SkillAccessError) as exc_info:
        await adapter.load("run-1", skill.name, catalog)

    assert "secret-token" not in str(exc_info.value)
    assert "private-bucket" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_resource_paths_are_validated_before_provider_access(
    skill: Skill,
) -> None:
    source = RecordingSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", skill.name, catalog)

    with pytest.raises(SkillUnsafePathError):
        await adapter.read_resource("run-1", skill.name, "../secret", catalog)
    with pytest.raises(SkillNotFoundError):
        await adapter.read_resource("run-1", skill.name, "missing.md", catalog)

    assert source.read_paths == []


@pytest.mark.asyncio
async def test_mismatched_provider_resource_is_rejected(skill: Skill) -> None:
    source = MismatchedResourceSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", skill.name, catalog)

    with pytest.raises(SkillMalformedError):
        await adapter.read_resource(
            "run-1", skill.name, "references/checklist.md", catalog
        )


@pytest.mark.asyncio
async def test_provider_resource_errors_and_malformed_values_are_safe(
    skill: Skill,
) -> None:
    for source, error_type in [
        (SkillErrorResourceSource(skill), SkillAccessError),
        (FailingResourceSource(skill), SkillAccessError),
        (MalformedResourceSource(skill), SkillMalformedError),
    ]:
        adapter = SkillRuntimeAdapter([source])
        catalog = await adapter.discover_catalog()
        await adapter.load("run-1", skill.name, catalog)
        with pytest.raises(error_type):
            await adapter.read_resource(
                "run-1", skill.name, "references/checklist.md", catalog
            )


@pytest.mark.asyncio
async def test_catalog_provenance_errors_reject_missing_or_legacy_indexes(
    skill: Skill,
) -> None:
    """Replay accepts old indexes but rejects missing or unknown provenance."""
    adapter = SkillRuntimeAdapter([InMemorySource(skill)])
    await adapter.discover_catalog()

    with pytest.raises(SkillNotFoundError):
        await adapter.load("run-1", skill.name, [], [])
    with pytest.raises(SkillCatalogError):
        await adapter.load(
            "run-1", skill.name, [{"name": skill.name, "_source_id": "missing"}]
        )
    with pytest.raises(SkillCatalogError):
        await adapter.load(
            "run-2", skill.name, [{"name": skill.name, "_source_index": 0}]
        )
    with pytest.raises(SkillCatalogError):
        await adapter.load(
            "run-3", skill.name, [{"name": skill.name, "_source_index": 99}]
        )


@pytest.mark.asyncio
async def test_invalid_utf8_resource_is_rejected(skill: Skill) -> None:
    binary_skill = skill.model_copy(
        update={
            "resources": {
                "invalid.txt": SkillResource(
                    path="invalid.txt", content=b"\xff", media_type="text/plain"
                )
            }
        }
    )
    adapter = SkillRuntimeAdapter([InMemorySource(binary_skill)])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", binary_skill.name, catalog)

    with pytest.raises(SkillUnsupportedMediaTypeError):
        await adapter.read_resource("run-1", binary_skill.name, "invalid.txt", catalog)


def test_adapter_rejects_invalid_source_values() -> None:
    with pytest.raises(TypeError, match="Skill or SkillSource"):
        SkillRuntimeAdapter([object()])


@pytest.mark.asyncio
async def test_malformed_provider_outputs_use_safe_errors(skill: Skill) -> None:
    with pytest.raises(SkillAccessError):
        await SkillRuntimeAdapter([SkillErrorMetadataSource()]).discover_catalog()
    with pytest.raises(SkillCatalogError):
        await SkillRuntimeAdapter([MalformedMetadataSource()]).discover_catalog()

    adapter = SkillRuntimeAdapter([MalformedLoadSource(skill)])
    catalog = await adapter.discover_catalog()
    with pytest.raises(SkillMalformedError):
        await adapter.load("run-1", skill.name, catalog)


@pytest.mark.asyncio
async def test_provider_load_errors_and_name_mismatches_are_safe(
    skill: Skill,
) -> None:
    for source, error_type in [
        (SkillErrorLoadSource(skill), SkillAccessError),
        (MismatchedLoadSource(skill), SkillMalformedError),
    ]:
        adapter = SkillRuntimeAdapter([source])
        catalog = await adapter.discover_catalog()
        with pytest.raises(error_type):
            await adapter.load("run-1", skill.name, catalog)


@pytest.mark.asyncio
async def test_discovery_fails_closed_when_one_source_fails(skill: Skill) -> None:
    with pytest.raises(SkillAccessError):
        await SkillRuntimeAdapter([skill, FailingMetadataSource()]).discover_catalog()


@pytest.mark.asyncio
async def test_duplicate_fallback_source_id_fails_closed(skill: Skill) -> None:
    first = InMemorySource(skill)
    second = InMemorySource(skill.model_copy(update={"name": "second-skill"}))

    with pytest.raises(SkillCatalogError):
        await SkillRuntimeAdapter([first, second]).discover_catalog()


@pytest.mark.asyncio
async def test_exact_duplicate_source_reference_is_deduplicated(skill: Skill) -> None:
    source = InMemorySource(skill)

    catalog = await SkillRuntimeAdapter([source, source]).discover_catalog()

    assert [entry["name"] for entry in catalog] == [skill.name]


@pytest.mark.parametrize("source_id", ["", "   ", 42])
def test_invalid_explicit_source_id_is_rejected(
    skill: Skill, source_id: object
) -> None:
    source = NamedSource(skill, source_id)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="source_id"):
        SkillRuntimeAdapter([source])


@pytest.mark.asyncio
async def test_source_identity_survives_source_reordering(skill: Skill) -> None:
    """Explicit source IDs keep persisted catalogs valid after reordering."""
    first_skill = skill.model_copy(update={"name": "first-skill"})
    second_skill = skill.model_copy(update={"name": "second-skill"})
    first = NamedSource(first_skill, "first")
    second = NamedSource(second_skill, "second")
    old_catalog = await SkillRuntimeAdapter([first, second]).discover_catalog()

    restarted = SkillRuntimeAdapter(
        [NamedSource(second_skill, "second"), NamedSource(first_skill, "first")]
    )
    loaded = await restarted.load("run-1", "first-skill", old_catalog)

    assert loaded.instructions == first_skill.instructions


@pytest.mark.asyncio
async def test_concurrent_duplicate_loads_are_single_flight(skill: Skill) -> None:
    source = BlockingSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()

    first = asyncio.create_task(adapter.load("run-1", skill.name, catalog))
    await source.started.wait()
    second = asyncio.create_task(adapter.load("run-1", skill.name, catalog))
    await asyncio.sleep(0)
    source.release.set()

    results = await asyncio.gather(first, second)

    assert results[0] == results[1]
    assert source.load_calls == 1


@pytest.mark.asyncio
async def test_resource_read_waits_for_inflight_load(skill: Skill) -> None:
    """A parallel resource read waits for activation to complete."""
    source = BlockingSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()

    load_task = asyncio.create_task(adapter.load("run-1", skill.name, catalog))
    await source.started.wait()
    read_task = asyncio.create_task(
        adapter.read_resource("run-1", skill.name, "references/checklist.md", catalog)
    )
    await asyncio.sleep(0)

    assert not read_task.done()
    source.release.set()

    loaded, resource = await asyncio.gather(load_task, read_task)

    assert loaded.name == skill.name
    assert resource.content == "# Checklist"
    assert source.load_calls == 1
    assert source.read_calls == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_resource_reads_are_single_flight(
    skill: Skill,
) -> None:
    """Concurrent readers should share one source read and one cache result."""
    source = BlockingResourceSource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", skill.name, catalog)

    first = asyncio.create_task(
        adapter.read_resource("run-1", skill.name, "references/checklist.md", catalog)
    )
    await source.started.wait()
    second = asyncio.create_task(
        adapter.read_resource("run-1", skill.name, "references/checklist.md", catalog)
    )
    await asyncio.sleep(0)
    source.release.set()

    results = await asyncio.gather(first, second)

    assert results[0] == results[1]
    assert source.read_calls == 1


@pytest.mark.asyncio
async def test_concurrent_resource_reads_enforce_run_budget() -> None:
    """Concurrent resource completion must still enforce the run-wide byte budget."""
    resources = {
        f"reference-{index}.txt": SkillResource(
            path=f"reference-{index}.txt",
            content=b"x" * MAX_RESOURCE_BYTES,
            media_type="text/plain",
        )
        for index in range((MAX_RUN_RESOURCE_BYTES // MAX_RESOURCE_BYTES) + 1)
    }
    skill = Skill(
        name="budgeted-skill",
        description="Budgeted resources.",
        instructions="Use resources.",
        resources=resources,
    )
    source = BlockingResourceSource(skill, expected_reads=len(resources))
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", skill.name, catalog)

    reads = [
        asyncio.create_task(adapter.read_resource("run-1", skill.name, path, catalog))
        for path in resources
    ]
    await source.started.wait()
    source.release.set()
    results = await asyncio.gather(*reads, return_exceptions=True)

    assert sum(isinstance(result, ReadSkillResourceResult) for result in results) == (
        MAX_RUN_RESOURCE_BYTES // MAX_RESOURCE_BYTES
    )
    assert sum(isinstance(result, SkillSizeLimitError) for result in results) == 1
    assert source.read_calls == len(resources)


@pytest.mark.asyncio
async def test_history_reconstruction_accepts_objects_and_ignores_malformed() -> None:
    """Only valid skill results rehydrate caches; malformed history is ignored."""
    skill = Skill(
        name="history-skill",
        description="History.",
        instructions="Use history.",
        resources={
            "reference.md": SkillResource(path="reference.md", content="history")
        },
    )
    source = InMemorySource(skill)
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    loaded = await adapter.load("run-1", skill.name, catalog)
    resource = await adapter.read_resource("run-1", skill.name, "reference.md", catalog)
    source.load_calls = 0
    source.read_calls = 0
    restored = SkillRuntimeAdapter([source])

    history = [
        SimpleNamespace(name="load_skill", content=loaded.model_dump_json()),
        SimpleNamespace(name="read_skill_resource", content=resource.model_dump_json()),
        {"name": "unknown_tool", "content": "{}"},
        {"name": "load_skill", "content": None},
        {"name": "read_skill_resource", "content": "not-json"},
    ]

    assert await restored.load("run-1", skill.name, catalog, history) == loaded
    assert (
        await restored.read_resource(
            "run-1", skill.name, "reference.md", catalog, history
        )
        == resource
    )
    assert source.load_calls == 0
    assert source.read_calls == 0


@pytest.mark.asyncio
async def test_history_reconstruction_keeps_same_paths_separate() -> None:
    first = Skill(
        name="first-skill",
        description="First.",
        instructions="First instructions.",
        resources={"reference.md": SkillResource(path="reference.md", content="first")},
    )
    second = Skill(
        name="second-skill",
        description="Second.",
        instructions="Second instructions.",
        resources={
            "reference.md": SkillResource(path="reference.md", content="second")
        },
    )
    adapter = SkillRuntimeAdapter([first, second])
    catalog = await adapter.discover_catalog()
    first_loaded = await adapter.load("run-1", first.name, catalog)
    second_loaded = await adapter.load("run-1", second.name, catalog)
    first_resource = await adapter.read_resource(
        "run-1", first.name, "reference.md", catalog
    )
    history = [
        {
            "role": "tool",
            "name": "load_skill",
            "content": first_loaded.model_dump_json(),
        },
        {
            "role": "tool",
            "name": "load_skill",
            "content": second_loaded.model_dump_json(),
        },
        {
            "role": "tool",
            "name": "read_skill_resource",
            "content": first_resource.model_dump_json(),
        },
    ]

    restored = SkillRuntimeAdapter([first, second])
    await restored.load("run-1", first.name, catalog, history)
    await restored.load("run-1", second.name, catalog, history)
    second_resource = await restored.read_resource(
        "run-1", second.name, "reference.md", catalog, history
    )

    assert second_resource.content == "second"


@pytest.mark.asyncio
async def test_duplicate_history_reads_count_once_toward_budget() -> None:
    resources = {
        f"reference-{index}.txt": SkillResource(
            path=f"reference-{index}.txt",
            content=b"x" * MAX_RESOURCE_BYTES,
            media_type="text/plain",
        )
        for index in range(4)
    }
    skill = Skill(
        name="budget-skill",
        description="Budget.",
        instructions="Use resources.",
        resources=resources,
    )
    adapter = SkillRuntimeAdapter([skill])
    catalog = await adapter.discover_catalog()
    loaded = await adapter.load("run-1", skill.name, catalog)
    first = await adapter.read_resource("run-1", skill.name, "reference-0.txt", catalog)
    history = [
        {"role": "tool", "name": "load_skill", "content": loaded.model_dump_json()},
        {
            "role": "tool",
            "name": "read_skill_resource",
            "content": first.model_dump_json(),
        },
        {
            "role": "tool",
            "name": "read_skill_resource",
            "content": first.model_dump_json(),
        },
    ]

    restored = SkillRuntimeAdapter([skill])
    await restored.load("run-1", skill.name, catalog, history)
    for path in resources:
        await restored.read_resource("run-1", skill.name, path, catalog, history)
