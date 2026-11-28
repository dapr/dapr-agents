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

import pytest
from pydantic import ValidationError

from dapr_agents.skills import (
    MAX_INSTRUCTIONS_BYTES,
    MAX_RESOURCE_BYTES,
    Skill,
    SkillMetadata,
    SkillResource,
)


def test_skill_metadata_keeps_stable_frontmatter_fields_top_level() -> None:
    metadata = SkillMetadata(
        name="incident-investigation",
        description="Investigate incidents.",
        license="Apache-2.0",
        compatibility="Requires read-only access.",
        metadata={"team": "sre"},
        extra_frontmatter={"x-owner": "platform"},
    )

    assert metadata.name == "incident-investigation"
    assert metadata.license == "Apache-2.0"
    assert metadata.compatibility == "Requires read-only access."
    assert metadata.metadata == {"team": "sre"}
    assert metadata.extra_frontmatter == {"x-owner": "platform"}


@pytest.mark.parametrize("field", ["name", "description"])
def test_skill_metadata_rejects_empty_required_frontmatter(field: str) -> None:
    values = {"name": "skill", "description": "description"}
    values[field] = ""

    with pytest.raises(ValidationError):
        SkillMetadata(**values)


@pytest.mark.parametrize("value", [1, True, ["sre"], {"team": "sre"}])
def test_skill_metadata_requires_string_values(value: object) -> None:
    with pytest.raises(ValidationError):
        SkillMetadata(
            name="skill",
            description="description",
            metadata={"owner": value},
        )


def test_skill_accepts_empty_instructions_and_valid_inline_resource() -> None:
    skill = Skill(
        name="incident-investigation",
        description="Investigate incidents.",
        instructions="",
        resources={
            "references/checklist.md": SkillResource(
                path="references/checklist.md",
                content="# Checklist",
                media_type="text/markdown",
            )
        },
    )

    assert skill.instructions == ""
    assert skill.resources["references/checklist.md"].content == "# Checklist"


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.md",
        "../escape.md",
        "references/../../escape.md",
        "bad\x00.md",
        "scripts/run.sh",
    ],
)
def test_skill_resource_rejects_unsafe_or_excluded_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        SkillResource(path=path, content="text", media_type="text/plain")


def test_skill_rejects_resource_key_that_does_not_match_resource_path() -> None:
    with pytest.raises(ValidationError):
        Skill(
            name="skill",
            description="description",
            instructions="instructions",
            resources={
                "references/one.md": SkillResource(
                    path="references/two.md",
                    content="text",
                    media_type="text/plain",
                )
            },
        )


def test_skill_rejects_instruction_content_over_byte_limit() -> None:
    with pytest.raises(ValidationError):
        Skill(
            name="skill",
            description="description",
            instructions="x" * (MAX_INSTRUCTIONS_BYTES + 1),
        )


def test_skill_resource_rejects_empty_media_type() -> None:
    with pytest.raises(ValidationError):
        SkillResource(path="reference.txt", content="text", media_type=" ")


def test_skill_resource_counts_utf8_bytes_and_rejects_over_limit() -> None:
    """Enforce the byte limit rather than the Python character count."""
    content = ("é" * (MAX_RESOURCE_BYTES // 2)) + "a"

    with pytest.raises(ValidationError):
        SkillResource(path="reference.txt", content=content, media_type="text/plain")


def test_skill_resource_accepts_bytes_for_source_materialization() -> None:
    """Sources may materialize bytes even though model reads are text-only."""
    resource = SkillResource(
        path="reference.bin",
        content=b"binary",
        media_type="application/octet-stream",
    )

    assert resource.content == b"binary"
