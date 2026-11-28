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

import dapr_agents
from dapr_agents import Skill, SkillMetadata, SkillResource, SkillSource
from dapr_agents.skills import __all__ as skills_all


def test_skill_types_are_reexported_from_package_root() -> None:
    assert {"Skill", "SkillMetadata", "SkillResource", "SkillSource"}.issubset(
        skills_all
    )
    assert {"Skill", "SkillMetadata", "SkillResource", "SkillSource"}.issubset(
        dapr_agents.__all__
    )
    # Keep internal runtime and tool-builder types out of the supported public API.
    assert not {
        "SkillCatalogEntry",
        "SkillRuntimeAdapter",
        "LoadSkillArguments",
        "ReadSkillResourceArguments",
        "build_skill_tools",
    }.intersection(skills_all)
    assert not {"SkillRuntimeAdapter", "build_skill_tools"}.intersection(
        dapr_agents.__all__
    )
    assert Skill.__name__ == "Skill"
    assert SkillMetadata.__name__ == "SkillMetadata"
    assert SkillResource.__name__ == "SkillResource"
    assert SkillSource.__name__ == "SkillSource"
