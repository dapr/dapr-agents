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
Trivial ``AgentExecutorBase`` implementation used by tests and as a
copy-pasteable skeleton for real executor providers.

``EchoAgentExecutor`` does no real reasoning — it echoes the prompt
back — but exercises the full event sequence (``text_delta`` ->
``message`` -> ``session`` -> ``complete``) that ``DurableAgent``'s
executor branch expects.
"""

from __future__ import annotations

import copy
from typing import Any, AsyncGenerator, Dict, List, Optional

from dapr_agents.agents.executors.base import AgentExecutorBase
from dapr_agents.agents.executors.event import AgentEvent

_DEFAULT_CHUNK = 16


class EchoAgentExecutor(AgentExecutorBase):
    """
    In-memory, zero-dependency executor that echoes the prompt.

    Args:
        chunk_size: Number of characters per ``text_delta`` event.
            Defaults to ``16``.
    """

    def __init__(self, *, chunk_size: int = _DEFAULT_CHUNK) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        sid = session_id or f"echo-{id(self):x}-{len(self._sessions)}"

        user_message = {"role": "user", "content": prompt}
        history = list(self._sessions.get(sid, ()))
        history.append(user_message)

        reply = f"echo: {prompt}"

        for start in range(0, len(reply), self._chunk_size):
            chunk = reply[start : start + self._chunk_size]
            yield AgentEvent(type="text_delta", content=chunk, session_id=sid)

        assistant_message = {"role": "assistant", "content": reply}
        history.append(assistant_message)
        self._sessions[sid] = history

        yield AgentEvent(
            type="message",
            content=copy.deepcopy(assistant_message),
            session_id=sid,
        )
        yield AgentEvent(
            type="session",
            content={"messages": copy.deepcopy(history)},
            session_id=sid,
        )
        yield AgentEvent(
            type="complete",
            content=copy.deepcopy(assistant_message),
            session_id=sid,
        )

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        history = self._sessions.get(session_id)
        if history is None:
            return None
        return {"messages": copy.deepcopy(history)}
