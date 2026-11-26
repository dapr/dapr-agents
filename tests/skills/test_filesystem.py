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

from pathlib import Path
from typing import Any

import pytest

from dapr_agents.skills import (
    MAX_INSTRUCTIONS_BYTES,
    MAX_RESOURCE_BYTES,
    FileSystemSkillSource,
    SkillAccessError,
    SkillCatalogError,
    SkillMalformedError,
    SkillNotFoundError,
    SkillSizeLimitError,
    SkillUnsafePathError,
)


def _write_skill(
    root: Path,
    name: str,
    *,
    frontmatter: str | None = None,
    instructions: str = "Follow the procedure.",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    metadata = frontmatter or f"name: {name}\ndescription: A useful skill."
    (skill_dir / "SKILL.md").write_text(
        f"---\n{metadata}\n---\n\n{instructions}", encoding="utf-8"
    )
    return skill_dir


@pytest.mark.parametrize("skill_paths", ["skills", Path("skills"), []])
def test_source_rejects_bare_or_empty_skill_paths(skill_paths: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        FileSystemSkillSource(skill_paths)


def test_source_rejects_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        FileSystemSkillSource([object()])
    with pytest.raises(ValueError):
        FileSystemSkillSource([tmp_path], max_depth=-1)
    with pytest.raises(ValueError):
        FileSystemSkillSource([tmp_path], max_depth=17)
    with pytest.raises(ValueError):
        FileSystemSkillSource([tmp_path], max_instructions_bytes=0)
    with pytest.raises(ValueError):
        FileSystemSkillSource(
            [tmp_path], max_instructions_bytes=MAX_INSTRUCTIONS_BYTES + 1
        )
    with pytest.raises(ValueError):
        FileSystemSkillSource([tmp_path], max_resource_bytes=MAX_RESOURCE_BYTES + 1)


@pytest.mark.asyncio
async def test_discovery_deduplicates_roots_and_obeys_depth(tmp_path: Path) -> None:
    _write_skill(tmp_path, "direct-skill")
    _write_skill(tmp_path / "group", "nested-skill")
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "not-a-skill" / "README.md").write_text("not a skill")

    source = FileSystemSkillSource([tmp_path, tmp_path], max_depth=1)

    metadata = await source.list_metadata()

    assert [item.name for item in metadata] == ["direct-skill"]


@pytest.mark.asyncio
async def test_discovery_supports_multiple_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "first-skill")
    _write_skill(second, "second-skill")

    source = FileSystemSkillSource([first, second])

    assert {item.name for item in await source.list_metadata()} == {
        "first-skill",
        "second-skill",
    }


@pytest.mark.asyncio
async def test_discovery_deduplicates_in_root_symlink_aliases(tmp_path: Path) -> None:
    _write_skill(tmp_path, "linked-skill")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    metadata = await FileSystemSkillSource([tmp_path]).list_metadata()

    assert [item.name for item in metadata] == ["linked-skill"]


@pytest.mark.asyncio
async def test_frontmatter_is_typed_and_unknown_fields_are_preserved(
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path,
        "incident-investigation",
        frontmatter=(
            "name: incident-investigation\n"
            "description: Investigate incidents.\n"
            "license: Apache-2.0\n"
            "compatibility: Requires read-only access.\n"
            "metadata:\n"
            "  team: sre\n"
            "allowed-tools: read_file\n"
            "x-owner: platform"
        ),
    )

    metadata = (await FileSystemSkillSource([tmp_path]).list_metadata())[0]

    assert metadata.license == "Apache-2.0"
    assert metadata.compatibility == "Requires read-only access."
    assert metadata.metadata == {"team": "sre"}
    assert metadata.extra_frontmatter == {
        "allowed-tools": "read_file",
        "x-owner": "platform",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frontmatter",
    [
        "description: missing name",
        "name: bad_name\ndescription: invalid name",
        "name: valid-name\ndescription:",
        "name: valid-name\ndescription: valid\nmetadata: [bad]",
        "- not-a-mapping",
        "name: valid-name\ndescription: [not a string]",
    ],
)
async def test_malformed_frontmatter_fails_closed(
    tmp_path: Path, frontmatter: str
) -> None:
    _write_skill(tmp_path, "valid-name", frontmatter=frontmatter)

    with pytest.raises(SkillMalformedError):
        await FileSystemSkillSource([tmp_path]).list_metadata()


@pytest.mark.asyncio
async def test_skill_directory_name_must_match_frontmatter_name(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "directory-name",
        frontmatter="name: metadata-name\ndescription: mismatch",
    )

    with pytest.raises(SkillMalformedError):
        await FileSystemSkillSource([tmp_path]).list_metadata()


@pytest.mark.asyncio
async def test_missing_frontmatter_fails_closed(tmp_path: Path) -> None:
    skill_dir = tmp_path / "missing-frontmatter"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("instructions only", encoding="utf-8")

    with pytest.raises(SkillMalformedError):
        await FileSystemSkillSource([tmp_path]).list_metadata()


@pytest.mark.asyncio
async def test_activation_and_resource_reads_are_lazy(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "lazy-skill", instructions="Use the guide.")
    resource_path = skill_dir / "docs" / "guide.data"
    resource_path.parent.mkdir()
    resource_path.write_bytes(b"exact bytes\x00")

    source = FileSystemSkillSource([tmp_path])
    await source.list_metadata()
    skill = await source.load_skill("lazy-skill")

    assert skill.instructions == "Use the guide."
    assert list(skill.resources) == ["docs/guide.data"]
    assert skill.resources["docs/guide.data"].content == b""

    resource = await source.read_resource("lazy-skill", "docs/guide.data")

    assert resource.content == b"exact bytes\x00"
    assert resource.media_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_empty_instructions_are_preserved(tmp_path: Path) -> None:
    _write_skill(tmp_path, "empty-skill", instructions="")

    skill = await FileSystemSkillSource([tmp_path]).load_skill("empty-skill")

    assert skill.instructions == ""


@pytest.mark.asyncio
async def test_instruction_limit_rejects_without_truncating(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "large-skill", instructions="x" * 100)
    document_size = (skill_dir / "SKILL.md").stat().st_size

    with pytest.raises(SkillSizeLimitError):
        await FileSystemSkillSource(
            [tmp_path], max_instructions_bytes=document_size - 1
        ).list_metadata()


@pytest.mark.asyncio
async def test_resource_limit_rejects_without_truncating(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "limited-skill")
    resource = skill_dir / "reference.txt"
    resource.write_bytes(b"12345")
    source = FileSystemSkillSource([tmp_path], max_resource_bytes=4)

    await source.list_metadata()
    with pytest.raises(SkillSizeLimitError):
        await source.read_resource("limited-skill", "reference.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/absolute.txt",
        "../escape.txt",
        "nested/../../escape.txt",
        ".hidden.txt",
        "nested/.hidden.txt",
        "bad\\\\path.txt",
        "bad\x00path.txt",
        "SKILL.md",
        "scripts/run.sh",
    ],
)
async def test_resource_paths_are_rejected_before_access(
    tmp_path: Path, path: str
) -> None:
    _write_skill(tmp_path, "safe-skill")
    source = FileSystemSkillSource([tmp_path])

    with pytest.raises(SkillUnsafePathError):
        await source.read_resource("safe-skill", path)


@pytest.mark.asyncio
async def test_missing_and_non_regular_resources_are_not_read(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "safe-skill")
    (skill_dir / "directory").mkdir()
    source = FileSystemSkillSource([tmp_path])

    with pytest.raises(SkillNotFoundError):
        await source.read_resource("safe-skill", "missing.txt")
    with pytest.raises(SkillNotFoundError):
        await source.read_resource("safe-skill", "directory")


@pytest.mark.asyncio
async def test_in_skill_symlink_is_allowed_but_escape_is_rejected(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill(tmp_path, "safe-skill")
    target = skill_dir / "target.txt"
    target.write_text("inside", encoding="utf-8")
    inside_link = skill_dir / "inside-link.txt"
    try:
        inside_link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    source = FileSystemSkillSource([tmp_path])
    assert (
        await source.read_resource("safe-skill", "inside-link.txt")
    ).content == b"inside"

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escape_link = skill_dir / "escape-link.txt"
    escape_link.symlink_to(outside)
    with pytest.raises(SkillUnsafePathError):
        await source.read_resource("safe-skill", "escape-link.txt")


@pytest.mark.asyncio
async def test_binary_bytes_remain_available_to_the_core_runtime(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill(tmp_path, "binary-skill")
    (skill_dir / "data.bin").write_bytes(b"\x00\xff")

    resource = await FileSystemSkillSource([tmp_path]).read_resource(
        "binary-skill", "data.bin"
    )

    assert resource.content == b"\x00\xff"
    assert resource.media_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_duplicate_skill_names_fail_at_source_discovery(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_skill(first, "same-skill")
    _write_skill(second, "same-skill")

    with pytest.raises(SkillCatalogError):
        await FileSystemSkillSource([first, second]).list_metadata()


@pytest.mark.asyncio
async def test_skill_document_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill_dir = root / "escaped-skill"
    skill_dir.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text(
        "---\nname: escaped-skill\ndescription: outside\n---\n\nsecret",
        encoding="utf-8",
    )
    try:
        (skill_dir / "SKILL.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(SkillUnsafePathError):
        await FileSystemSkillSource([root]).list_metadata()


@pytest.mark.asyncio
async def test_external_resource_symlink_is_rejected_during_discovery(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill(tmp_path, "safe-skill")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (skill_dir / "reference.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(SkillUnsafePathError):
        await FileSystemSkillSource([tmp_path]).list_metadata()


@pytest.mark.asyncio
async def test_persisted_snapshot_reopens_with_a_new_source_instance(
    tmp_path: Path,
) -> None:
    skill_dir = _write_skill(tmp_path, "replayable-skill")
    resource = skill_dir / "reference.txt"
    resource.write_text("stable", encoding="utf-8")

    from dapr_agents.skills._runtime import SkillRuntimeAdapter

    catalog = await SkillRuntimeAdapter(
        [FileSystemSkillSource([tmp_path])]
    ).discover_catalog()
    restarted = SkillRuntimeAdapter([FileSystemSkillSource([tmp_path])])

    loaded = await restarted.load("run-1", "replayable-skill", catalog)
    read = await restarted.read_resource(
        "run-1", "replayable-skill", "reference.txt", catalog
    )

    assert loaded.available_resources == ["reference.txt"]
    assert read.content == "stable"


@pytest.mark.asyncio
async def test_snapshot_reopens_and_rejects_unavailable_source(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "replayable-skill")
    source = FileSystemSkillSource([tmp_path])
    snapshot_id = await source.resolve_snapshot()

    restarted = FileSystemSkillSource([tmp_path])
    reopened = await restarted.resolve_snapshot(snapshot_id=snapshot_id)
    metadata = await restarted.list_metadata(snapshot_id=reopened)

    assert [item.name for item in metadata] == ["replayable-skill"]

    with pytest.raises(SkillCatalogError):
        await FileSystemSkillSource([tmp_path / "unavailable"]).resolve_snapshot(
            snapshot_id=snapshot_id
        )


@pytest.mark.asyncio
async def test_resource_mutation_after_snapshot_fails_closed(tmp_path: Path) -> None:
    skill_dir = _write_skill(tmp_path, "stable-skill")
    resource = skill_dir / "reference.txt"
    resource.write_text("initial", encoding="utf-8")

    from dapr_agents.skills._runtime import SkillRuntimeAdapter

    source = FileSystemSkillSource([tmp_path])
    adapter = SkillRuntimeAdapter([source])
    catalog = await adapter.discover_catalog()
    await adapter.load("run-1", "stable-skill", catalog)
    resource.write_text("changed", encoding="utf-8")

    with pytest.raises(SkillAccessError):
        await adapter.read_resource("run-1", "stable-skill", "reference.txt", catalog)
