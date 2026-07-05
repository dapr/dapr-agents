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

import pytest

from dapr_agents.agents.configs import AgentExecutionConfig, BuiltinTool


def test_builtin_tools_accepts_enum_member() -> None:
    cfg = AgentExecutionConfig(streaming=True, builtin_tools=[BuiltinTool.ASK_USER])
    # Normalized to the plain string value so downstream lookups are uniform.
    assert cfg.builtin_tools == ["ask_user"]
    assert all(isinstance(name, str) for name in cfg.builtin_tools)


def test_builtin_tools_accepts_raw_string() -> None:
    cfg = AgentExecutionConfig(streaming=True, builtin_tools=["ask_user"])
    assert cfg.builtin_tools == ["ask_user"]


def test_builtin_tools_accepts_mixed_and_defaults_empty() -> None:
    assert AgentExecutionConfig().builtin_tools == []
    cfg = AgentExecutionConfig(
        streaming=True, builtin_tools=[BuiltinTool.ASK_USER, "ask_user"]
    )
    assert cfg.builtin_tools == ["ask_user", "ask_user"]


def test_builtin_tools_rejects_unknown_name() -> None:
    with pytest.raises(ValueError) as exc_info:
        AgentExecutionConfig(streaming=True, builtin_tools=["not_a_tool"])
    message = str(exc_info.value)
    assert "not_a_tool" in message
    # The error should surface the valid choices so the fix is obvious.
    assert "ask_user" in message


def test_factory_registry_covers_every_builtin_tool() -> None:
    # Guard against enum/registry drift: every BuiltinTool member must have a
    # factory, otherwise a validated config would still fail to register.
    from dapr_agents.agents.durable import BUILTIN_TOOL_FACTORIES

    for tool in BuiltinTool:
        assert tool in BUILTIN_TOOL_FACTORIES, (
            f"{tool!r} has no factory in BUILTIN_TOOL_FACTORIES"
        )
