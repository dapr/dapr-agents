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

import re
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBytes,
    StrictStr,
    field_validator,
    model_validator,
)

MAX_INSTRUCTIONS_BYTES = 64 * 1024
MAX_RESOURCE_BYTES = 1024 * 1024
MAX_RUN_RESOURCE_BYTES = 4 * 1024 * 1024

_MAX_SKILL_NAME_LENGTH = 64
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _content_size(content: str | bytes) -> int:
    return len(content.encode("utf-8")) if isinstance(content, str) else len(content)


def _validate_skill_path(path: str) -> str:
    """Validate a resource path as a relative POSIX exact path.

    Glob patterns are unsupported.
    """
    if (
        not path
        or "\x00" in path
        or path.startswith(("/", "\\"))
        or ":" in path.split("/", 1)[0]
        or "\\" in path
        or any(character in path for character in "*?[]")
    ):
        raise ValueError("unsafe skill resource path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or parts[0] == "scripts":
        raise ValueError("unsafe skill resource path")
    return path


class SkillMetadata(BaseModel):
    """Canonical Agent Skills metadata used to advertise catalog entries."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(
        description=(
            "Canonical Agent Skills name: 1-64 lowercase letters, numbers, and "
            "hyphens, without leading, trailing, or consecutive hyphens."
        )
    )
    description: StrictStr = Field(
        description=(
            "Required description of what the skill does and when to use it; "
            "must contain at least one non-whitespace character."
        )
    )
    license: StrictStr | None = Field(
        default=None,
        description="Optional license name or reference to a bundled license file.",
    )
    compatibility: StrictStr | None = Field(
        default=None,
        description="Optional environment requirements for using the skill.",
    )
    metadata: dict[str, StrictStr] = Field(
        default_factory=dict,
        description=(
            "Optional map of string keys to string values for additional metadata."
        ),
    )
    extra_frontmatter: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Unknown or experimental frontmatter fields are preserved for forward "
            "compatibility; they are not interpreted by the current runtime."
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if (
            not value
            or len(value) > _MAX_SKILL_NAME_LENGTH
            or not _SKILL_NAME_PATTERN.fullmatch(value)
        ):
            raise ValueError(
                f"name must be 1-{_MAX_SKILL_NAME_LENGTH} characters of lowercase "
                "letters, numbers, and single hyphens"
            )
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "description must contain at least one non-whitespace character"
            )
        return value


class SkillResource(BaseModel):
    """Resource content and media type for an Agent Skill."""

    model_config = ConfigDict(extra="forbid")

    path: StrictStr = Field(
        description=(
            "Relative POSIX path identifying the resource within the skill; "
            "absolute paths, traversal, and scripts/ paths are rejected."
        )
    )
    content: StrictStr | StrictBytes = Field(
        description="Resource content represented as text or bytes."
    )
    media_type: StrictStr = Field(
        default="text/plain",
        description=(
            "Media type describing the resource content; defaults to "
            "text/plain when unspecified."
        ),
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_skill_path(value)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("media_type must not be empty")
        return value

    @model_validator(mode="after")
    def validate_size(self) -> "SkillResource":
        if _content_size(self.content) > MAX_RESOURCE_BYTES:
            raise ValueError("skill resource exceeds the maximum size")
        return self


class Skill(SkillMetadata):
    """A directly configured Agent Skill with inline instructions and resources."""

    instructions: StrictStr = Field(
        description=(
            "Skill instructions returned by load_skill; they may be empty and do "
            "not modify the agent's system prompt."
        )
    )
    resources: dict[str, SkillResource] = Field(
        default_factory=dict,
        description=(
            "Inline resources keyed by validated relative paths; each key must match "
            "the corresponding resource path."
        ),
    )

    @model_validator(mode="after")
    def validate_content(self) -> "Skill":
        if _content_size(self.instructions) > MAX_INSTRUCTIONS_BYTES:
            raise ValueError("skill instructions exceed the maximum size")
        for resource_path, resource in self.resources.items():
            if resource_path != resource.path:
                raise ValueError("skill resource key must match its path")
        return self


__all__ = [
    "MAX_INSTRUCTIONS_BYTES",
    "MAX_RESOURCE_BYTES",
    "MAX_RUN_RESOURCE_BYTES",
    "SkillMetadata",
    "SkillResource",
    "Skill",
]
