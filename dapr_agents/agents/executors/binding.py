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

"""
What a hosting agent offers an ``AgentExecutorBase``.

``DurableAgent`` builds an ``ExecutorBinding`` for every executor run and
passes it to ``AgentExecutorBase.bind``. Executors use whatever parts they
support (the agent's system prompt, tools, hooks, a durable session store)
and ignore the rest; the default ``bind`` ignores all of it, so existing
executors keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from dapr_agents.hooks import BeforeToolHook
from dapr_agents.tool.base import AgentTool


@dataclass(frozen=True)
class ExecutorBinding:
    """
    The hosting agent's configuration for one executor run.

    Values set explicitly on the executor's own configuration should win
    over the binding; the binding fills in what the executor left unset.

    Attributes:
        agent_name: Name of the hosting agent.
        system_prompt: The agent's rendered system prompt (profile role,
            goal, instructions, or an explicit ``system_prompt``).
        max_iterations: The agent's ``AgentExecutionConfig.max_iterations``,
            a reasonable cap for the executor's own turn limit.
        tools: The agent's tools, ready to call from inside the activity
            that drives the executor (workflow-backed tools such as
            agents-as-tools and Dapr MCPServer tools are already bridged).
        before_tool_call: The agent's ``before_tool_call`` hooks. Executors
            that support tool approval map ``RequireApproval`` onto a
            ``paused`` run; see ``AgentExecutorBase.supports_tool_approval``.
        session_store: A durable session store offered by the host (a
            ``DaprSessionStore`` on the agent's state store), or ``None``.
    """

    agent_name: str
    system_prompt: Optional[str] = None
    max_iterations: Optional[int] = None
    tools: Tuple[AgentTool, ...] = ()
    before_tool_call: Tuple[BeforeToolHook, ...] = ()
    session_store: Optional[Any] = None
