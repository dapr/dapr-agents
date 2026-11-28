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
    """Async source for discovering and reading Agent Skills.

    Implementations return skill models and may raise public ``SkillError``
    subclasses. Temporary source failures should raise
    ``SkillTransientAccessError`` so the durable activity retry policy can
    retry them. Unexpected exceptions are classified and redacted by the
    runtime.

    The configuration that adds a source owns its non-empty ``source_id``
    value. The source instance may receive it through its constructor or
    expose it as a configured attribute. Use it whenever multiple instances
    of the same source implementation class are configured, or when replay
    must survive source implementation renames or moves. The ID is a stable,
    non-secret logical source identity; it must remain unchanged for active
    or replayable workflows. Sources without an explicit ID use the qualified
    source implementation class name as a compatibility fallback, and duplicate
    fallback identities fail closed. Source methods are not model-facing tools.

    Every operation must honor the supplied snapshot ID and must not silently
    substitute a different snapshot.
    """

    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        """Resolve and validate a skill snapshot ID.

        ``None`` selects the current snapshot. A supplied ID must be validated
        and either returned as a usable, non-empty snapshot ID or rejected.

        Raises:
            SkillCatalogError: If the snapshot ID is invalid or unavailable.
            SkillTransientAccessError: If resolution temporarily fails.
            SkillAccessError: If resolution permanently fails.
        """
        ...

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        """Return skill metadata from the selected skill catalog.

        Return an empty sequence when the skill catalog contains no
        discoverable skills. Resource inventories may remain unavailable and
        be resolved lazily later.

        Raises:
            SkillCatalogError: If the skill catalog is invalid or malformed.
            SkillTransientAccessError: If discovery temporarily fails.
            SkillAccessError: If discovery permanently fails.
        """
        ...

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        """Load one skill's instructions and resource metadata.

        The returned skill name must match ``name``. Resource contents may be
        omitted when the source supports lazy resource loading.

        Raises:
            SkillNotFoundError: If the named skill is absent.
            SkillMalformedError: If the skill data is invalid.
            SkillTransientAccessError: If loading the skill temporarily fails.
            SkillAccessError: If loading the skill permanently fails.
        """
        ...

    async def read_resource(
        self,
        name: str,
        path: str,
        *,
        snapshot_id: str | None = None,
    ) -> SkillResource:
        """Read the content of one resource belonging to a named skill.

        The returned resource path must match ``path`` and belong to ``name``.
        The source must validate the path and honor the selected snapshot.

        Raises:
            SkillNotFoundError: If the skill or resource is absent.
            SkillUnsafePathError: If ``path`` is unsafe.
            SkillSizeLimitError: If the resource exceeds the configured limit.
            SkillUnsupportedMediaTypeError: If the resource cannot be read as
                a supported resource type.
            SkillTransientAccessError: If reading the skill resource
                temporarily fails.
            SkillAccessError: If reading the skill resource permanently fails.
        """
        ...


__all__ = ["SkillSource"]
