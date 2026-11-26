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

from typing import Sequence

from dapr_agents.skills import (
    Skill,
    SkillMetadata,
    SkillResource,
    SkillSource,
)


class FakeSkillSource:
    async def resolve_snapshot(self, *, snapshot_id: str | None = None) -> str:
        return "fake"

    async def list_metadata(
        self, *, snapshot_id: str | None = None
    ) -> Sequence[SkillMetadata]:
        return []

    async def load_skill(self, name: str, *, snapshot_id: str | None = None) -> Skill:
        raise AssertionError(name)

    async def read_resource(
        self, name: str, path: str, *, snapshot_id: str | None = None
    ) -> SkillResource:
        raise AssertionError((name, path))


def test_skill_source_is_a_runtime_checkable_public_protocol() -> None:
    assert isinstance(FakeSkillSource(), SkillSource)
