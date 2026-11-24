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

from typing import Optional


class SkillError(Exception):
    """Base class for safe, stable public Agent Skills errors."""

    category = "skill"

    def __init__(
        self,
        skill_name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.skill_name = skill_name
        self.path = path
        context = []
        if skill_name is not None:
            context.append(f"skill={skill_name!r}")
        if path is not None:
            context.append(f"path={path!r}")
        suffix = f" ({', '.join(context)})" if context else ""
        super().__init__(f"Skill {self.category} error{suffix}")


class SkillNotFoundError(SkillError):
    category = "not found"


class SkillMalformedError(SkillError):
    category = "malformed"


class SkillUnsafePathError(SkillError):
    category = "unsafe path"


class SkillAccessError(SkillError):
    category = "access"


class SkillUnsupportedMediaTypeError(SkillError):
    category = "unsupported media type"


class SkillSizeLimitError(SkillError):
    category = "size limit"


class SkillCatalogError(SkillError):
    category = "catalog"


__all__ = [
    "SkillError",
    "SkillNotFoundError",
    "SkillMalformedError",
    "SkillUnsafePathError",
    "SkillAccessError",
    "SkillUnsupportedMediaTypeError",
    "SkillSizeLimitError",
    "SkillCatalogError",
]
