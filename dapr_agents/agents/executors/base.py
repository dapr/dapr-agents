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
Abstractions for stateful agent runtimes.

Where ``ChatClientBase`` models the stateless function
``messages -> completion``, ``AgentExecutorBase`` models runtimes that
own the full agent loop — session state, tool dispatch, multi-turn
reasoning — and emit typed ``AgentEvent`` values as an async stream.

Typical integration targets include the Claude Agent SDK, LangGraph,
AutoGen, and the OpenAI Assistants API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, ClassVar, Dict, Optional

from dapr_agents.agents.executors.binding import ExecutorBinding
from dapr_agents.agents.executors.event import AgentEvent


class AgentExecutorBase(ABC):
    """
    Base class for LLM integrations that manage the full agent loop.

    Unlike ``ChatClientBase`` (stateless completion), an
    ``AgentExecutorBase`` maintains session state and yields typed events.
    Implementations are typically thin wrappers around an external
    runtime (e.g. ``claude_agent_sdk.query``, LangGraph
    ``astream_events``).

    The contract is intentionally narrow so that consumers such as
    ``DurableAgent`` can drive any compliant executor without
    provider-specific branching.

    Executors that can pause a run on a tool call awaiting a decision set
    ``supports_tool_approval = True``; see ``run`` and the module docs of
    ``dapr_agents.agents.executors.event`` for the ``paused`` contract.

    Executors that can use the hosting agent's profile, tools, hooks or
    session store override ``bind``; the default ignores the binding.
    """

    supports_tool_approval: ClassVar[bool] = False
    """Whether ``run`` may end with a ``paused`` event and accepts
    ``context[CONTEXT_TOOL_DECISIONS]`` to resume it."""

    def bind(self, binding: ExecutorBinding) -> "AgentExecutorBase":
        """
        Return an executor configured with what the hosting agent offers.

        ``DurableAgent`` calls this before every run with the agent's
        current system prompt, tools, ``before_tool_call`` hooks and
        durable session store, and drives the returned executor. The
        default returns ``self`` unchanged. Overrides must not mutate
        ``self`` (return a new instance instead) and should let settings
        made explicitly on the executor win over the binding.

        Args:
            binding: The hosting agent's configuration for this run.

        Returns:
            The executor to drive for this run.
        """
        return self

    @abstractmethod
    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Execute a single agent turn, yielding events in order.

        Args:
            prompt: User/task input for this turn.
            session_id: Optional identifier to resume a prior session.
                If ``None``, the executor should create a fresh session.
            context: Provider-specific extras (MCP server handles,
                tool catalogs, scoped permissions, etc.). Implementations
                may ignore unknown keys. Executors that support tool
                approval read ``context[CONTEXT_TOOL_DECISIONS]`` to resume
                a paused run; the prompt is then ignored.

        Yields:
            ``AgentEvent`` values. The stream MUST end with either a
            ``complete`` event (success) or an ``error`` event (terminal
            failure). Executors with ``supports_tool_approval`` may instead
            end with a ``paused`` event (a deferred tool call awaits a
            decision). Consumers may treat absence of a terminal event as
            an executor bug.
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the persisted state for ``session_id``.

        Args:
            session_id: Identifier previously emitted by ``run``.

        Returns:
            A provider-defined snapshot of the session
            (typically ``{"messages": [...], "metadata": {...}}``),
            or ``None`` if the executor has no record of it.
        """
        ...
