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

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictStr

from dapr_agents.skills._constants import (
    LOAD_SKILL_TOOL_NAME,
    READ_SKILL_RESOURCE_TOOL_NAME,
)
from dapr_agents.skills.errors import SkillAccessError
from dapr_agents.tool.base import AgentTool


class LoadSkillArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr


class ReadSkillResourceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: StrictStr
    path: StrictStr


def _skill_tool_placeholder(**_: Any) -> None:
    raise SkillAccessError()


def build_skill_tools() -> tuple[AgentTool, AgentTool]:
    """Build the model-facing tools routed by the durable skill runtime."""
    return (
        AgentTool(
            name=LOAD_SKILL_TOOL_NAME,
            description="Activate a skill and return its instructions and resources.",
            args_model=LoadSkillArguments,
            func=_skill_tool_placeholder,
        ),
        AgentTool(
            name=READ_SKILL_RESOURCE_TOOL_NAME,
            description="Read a UTF-8 text resource from an activated skill.",
            args_model=ReadSkillResourceArguments,
            func=_skill_tool_placeholder,
        ),
    )
