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

"""Tests for ``ClaudeAgentExecutorConfig`` (importable without the SDK)."""

import pytest

from dapr_agents.agents.executors import ClaudeAgentExecutorConfig, ExecutorBinding
from dapr_agents.tool import tool


@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


@tool
def transfer(to: str) -> str:
    """Transfer money."""
    return f"sent to {to}"


def _hook(ctx):
    return None


def _other_hook(ctx):
    return None


class TestValidation:
    def test_sequences_are_stored_as_tuples(self):
        config = ClaudeAgentExecutorConfig(
            allowed_tools=["a"], tools=[get_weather], setting_sources=["user"]
        )
        assert config.allowed_tools == ("a",)
        assert config.tools == (get_weather,)
        assert config.setting_sources == ("user",)

    def test_builtin_tools_none_keeps_cli_default(self):
        assert ClaudeAgentExecutorConfig(builtin_tools=None).builtin_tools is None

    def test_defaults_are_locked_down(self):
        config = ClaudeAgentExecutorConfig()
        assert config.builtin_tools == ()
        assert config.setting_sources == ()
        assert config.include_partial_messages is True

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_turns": 0}, "max_turns"),
            ({"tool_server_name": ""}, "tool_server_name"),
            (
                {"tools": [get_weather], "mcp_servers": {"dapr": {}}},
                "mcp_servers already defines",
            ),
            ({"extra_options": {"resume": "x", "hooks": {}}}, "hooks, resume"),
            ({"extra_options": {"model": "x"}}, r"model \(use model\)"),
            ({"extra_options": {"tools": []}}, r"tools \(use builtin_tools\)"),
            (
                {"extra_options": {"session_store": object()}},
                r"session_store \(use session_store\)",
            ),
        ],
    )
    def test_rejects_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ClaudeAgentExecutorConfig(**kwargs)

    def test_tool_names_use_server_prefix(self):
        config = ClaudeAgentExecutorConfig(tools=[get_weather], tool_server_name="t")
        assert config.tool_names() == (f"mcp__t__{get_weather.name}",)


class TestBoundTo:
    def test_fills_unset_values_from_binding(self):
        binding = ExecutorBinding(
            agent_name="a",
            system_prompt="SYS",
            max_iterations=7,
            tools=(get_weather,),
            before_tool_call=(_hook,),
            session_store="store",
        )
        bound = ClaudeAgentExecutorConfig(model="m").bound_to(binding)
        assert bound.system_prompt == "SYS"
        assert bound.max_turns == 7
        assert bound.session_store == "store"
        assert bound.tools == (get_weather,)
        assert bound.before_tool_call == (_hook,)
        assert bound.model == "m"

    def test_explicit_settings_win(self):
        binding = ExecutorBinding(
            agent_name="a",
            system_prompt="SYS",
            max_iterations=7,
            session_store="store",
        )
        config = ClaudeAgentExecutorConfig(
            system_prompt="MINE", max_turns=2, session_store="own"
        )
        bound = config.bound_to(binding)
        assert (bound.system_prompt, bound.max_turns, bound.session_store) == (
            "MINE",
            2,
            "own",
        )

    def test_merges_tools_and_hooks_without_duplicates(self):
        config = ClaudeAgentExecutorConfig(
            tools=[get_weather], before_tool_call=[_hook]
        )
        binding = ExecutorBinding(
            agent_name="a",
            tools=(get_weather, transfer),
            before_tool_call=(_hook, _other_hook),
        )
        bound = config.bound_to(binding)
        assert [t.name for t in bound.tools] == [get_weather.name, transfer.name]
        assert bound.before_tool_call == (_hook, _other_hook)

    def test_does_not_mutate_original(self):
        config = ClaudeAgentExecutorConfig()
        config.bound_to(ExecutorBinding(agent_name="a", system_prompt="S"))
        assert config.system_prompt is None
