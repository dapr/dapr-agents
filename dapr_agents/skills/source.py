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

from typing import Protocol, Sequence, runtime_checkable

from dapr_agents.skills.models import Skill, SkillMetadata, SkillResource


@runtime_checkable
class SkillSource(Protocol):
    """Async full-lifecycle source protocol for Agent Skills.

    The configuration that adds a source owns its non-empty ``source_id``
    value. The source instance may receive it through its constructor or
    expose it as a configured attribute. Use it whenever multiple instances
    of the same source implementation class are configured, or when replay
    must survive source implementation renames or moves. The ID is a stable,
    non-secret logical source identity; it must remain unchanged for active
    or replayable workflows. Sources without an explicit ID use the qualified
    source implementation class name as a compatibility fallback, and duplicate
    fallback identities fail closed. Source methods are not model-facing tools.
    """

    async def list_metadata(self) -> Sequence[SkillMetadata]:
        """Return catalog metadata for skills available from this source."""
        ...

    async def load_skill(self, name: str) -> Skill:
        """Load instructions and resource metadata for the named skill."""
        ...

    async def read_resource(self, name: str, path: str) -> SkillResource:
        """Read one resource belonging to the named skill."""
        ...


__all__ = ["SkillSource"]
