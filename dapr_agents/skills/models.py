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
from typing import Any, Dict, Optional

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

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _content_size(content: str | bytes) -> int:
    return len(content.encode("utf-8")) if isinstance(content, str) else len(content)


def _validate_skill_path(path: str) -> str:
    """Validate a resource path as a safe relative POSIX path."""
    if (
        not path
        or "\x00" in path
        or path.startswith(("/", "\\"))
        or ":" in path.split("/", 1)[0]
        or "\\" in path
    ):
        raise ValueError("unsafe skill resource path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts) or parts[0] == "scripts":
        raise ValueError("unsafe skill resource path")
    return path


class SkillMetadata(BaseModel):
    """Canonical Agent Skills metadata used to advertise catalog entries."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    description: StrictStr
    license: Optional[StrictStr] = None
    compatibility: Optional[StrictStr] = None
    metadata: Dict[str, StrictStr] = Field(default_factory=dict)
    extra_frontmatter: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or len(value) > 64 or not _SKILL_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "name must be 1-64 characters of lowercase letters, numbers, and single hyphens"
            )
        return value

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty")
        return value


class SkillResource(BaseModel):
    """Resource content and media type for an Agent Skill."""

    model_config = ConfigDict(extra="forbid")

    path: StrictStr
    content: StrictStr | StrictBytes
    media_type: StrictStr = "text/plain"

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

    instructions: StrictStr
    resources: Dict[str, SkillResource] = Field(default_factory=dict)

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
