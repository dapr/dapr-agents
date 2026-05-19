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

"""Unit tests for ``EchoAgentExecutor``."""

import pytest

from dapr_agents.agents.executors import AgentEvent, EchoAgentExecutor


async def _collect(executor, prompt, *, session_id=None):
    """Drain the async generator into a list for assertions."""
    events = []
    async for event in executor.run(prompt, session_id=session_id):
        events.append(event)
    return events


class TestEchoAgentExecutor:
    @pytest.mark.asyncio
    async def test_emits_terminal_complete_event(self):
        executor = EchoAgentExecutor()
        events = await _collect(executor, "hello", session_id="s1")

        assert events[-1].type == "complete"
        assert isinstance(events[-1].content, dict)
        assert events[-1].content["role"] == "assistant"
        assert events[-1].content["content"] == "echo: hello"
        assert events[-1].session_id == "s1"

    @pytest.mark.asyncio
    async def test_emits_expected_event_sequence(self):
        executor = EchoAgentExecutor(chunk_size=4)
        events = await _collect(executor, "abcdefgh", session_id="s2")

        types = [e.type for e in events]
        # Deltas first, then message, session, complete.
        assert types[-3:] == ["message", "session", "complete"]
        assert types[:-3] == ["text_delta"] * (len(types) - 3)
        assert all(e.session_id == "s2" for e in events)

    @pytest.mark.asyncio
    async def test_session_id_auto_assigned(self):
        executor = EchoAgentExecutor()
        events = await _collect(executor, "x")
        assert events[-1].session_id is not None
        assert events[-1].session_id.startswith("echo-")

    @pytest.mark.asyncio
    async def test_get_session_returns_history(self):
        executor = EchoAgentExecutor()
        await _collect(executor, "first", session_id="s3")
        await _collect(executor, "second", session_id="s3")

        snapshot = await executor.get_session("s3")
        assert snapshot is not None
        assert snapshot["messages"][0] == {"role": "user", "content": "first"}
        assert snapshot["messages"][1] == {
            "role": "assistant",
            "content": "echo: first",
        }
        assert snapshot["messages"][2] == {"role": "user", "content": "second"}
        assert snapshot["messages"][3] == {
            "role": "assistant",
            "content": "echo: second",
        }

    @pytest.mark.asyncio
    async def test_get_session_unknown_returns_none(self):
        executor = EchoAgentExecutor()
        assert await executor.get_session("does-not-exist") is None

    @pytest.mark.asyncio
    async def test_session_snapshot_is_isolated_from_internal_state(self):
        executor = EchoAgentExecutor()
        await _collect(executor, "hi", session_id="s4")
        snapshot = await executor.get_session("s4")
        assert snapshot is not None
        snapshot["messages"].append({"role": "user", "content": "tampered"})
        fresh = await executor.get_session("s4")
        assert fresh is not None
        assert len(fresh["messages"]) == 2

    @pytest.mark.asyncio
    async def test_all_events_are_agent_events(self):
        executor = EchoAgentExecutor()
        events = await _collect(executor, "ping")
        assert all(isinstance(e, AgentEvent) for e in events)

    def test_invalid_chunk_size_rejected(self):
        with pytest.raises(ValueError):
            EchoAgentExecutor(chunk_size=0)
