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

"""Tests for the history-derived conversation path.

``test_durable_executor`` holds the old path's copy; at the contract release
the old copy goes and this module takes its name.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import Mock
import pytest
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from tests.agents.durableagent.test_durable_agent import _activity_method_name
from dapr_agents.agents.executors import (
    AgentEvent,
    AgentExecutorBase,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types import AgentError
from tests.agents.durableagent.test_durable_executor import (  # noqa: F401
    patch_dapr_check,
    MockDaprClient,
    setup_env,
    _make_agent,
    _ScriptedExecutor,
)

pytestmark = pytest.mark.usefixtures("history_path_only")


def _consume(agent: DurableAgent, payload: Dict[str, Any]) -> Any:
    return asyncio.run(agent._consume_executor(payload))


def _completes(content: str = "done") -> List[AgentEvent]:
    return [
        AgentEvent(type="complete", content={"role": "assistant", "content": content})
    ]


class TestConsumeExecutor:
    """Unit-tests the async consumer that the run_executor activity drives."""

    def test_returns_final_message(self):
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="message",
                    content={"role": "assistant", "content": "reply"},
                    session_id="s",
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "reply"},
                    session_id="s",
                ),
            ]
        )
        agent = _make_agent(executor)

        result = _consume(
            agent,
            {
                "task": "hi",
                "instance_id": "inst-1",
                "session_id": "s",
                "source": "test",
            },
        )

        assert result == {"role": "assistant", "content": "reply"}
        assert executor.calls == [{"prompt": "hi", "session_id": "s", "context": None}]

    def test_error_event_raises_agent_error(self):
        agent = _make_agent(
            _ScriptedExecutor([AgentEvent(type="error", content="boom")])
        )
        with pytest.raises(AgentError, match="boom"):
            _consume(agent, {"task": "hi", "instance_id": "inst-1"})

    def test_missing_complete_event_raises(self):
        executor = _ScriptedExecutor(
            [AgentEvent(type="message", content={"role": "assistant", "content": "x"})]
        )
        with pytest.raises(AgentError, match="without a 'complete' event"):
            _consume(_make_agent(executor), {"task": "hi", "instance_id": "inst-1"})

    def test_caller_session_id_passes_through(self):
        """Caller-supplied session_id must reach executor.run unchanged."""
        executor = _ScriptedExecutor(_completes())
        _consume(
            _make_agent(executor),
            {"task": "hi", "instance_id": "inst-1", "session_id": "sess-resume"},
        )
        assert executor.calls[-1]["session_id"] == "sess-resume"

    def test_omitted_session_id_yields_none_to_executor(self):
        """No payload session_id: executor.run gets None so it can auto-assign."""
        executor = _ScriptedExecutor(_completes())
        _consume(_make_agent(executor), {"task": "hi", "instance_id": "inst-1"})
        assert executor.calls[-1]["session_id"] is None

    def test_caller_context_threads_to_executor(self):
        """Caller-supplied context dict must reach executor.run unchanged."""
        ctx_payload = {"mcp_servers": ["primary"], "scopes": ["read"]}
        executor = _ScriptedExecutor(_completes())
        _consume(
            _make_agent(executor),
            {"task": "hi", "instance_id": "inst-1", "context": ctx_payload},
        )
        assert executor.calls[-1]["context"] == ctx_payload

    def test_executor_exception_becomes_agent_error(self):
        """A generic exception mid-stream surfaces as an AgentError."""

        class _BoomExecutor(AgentExecutorBase):
            async def run(
                self,
                prompt: str,
                *,
                session_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> AsyncIterator[AgentEvent]:
                yield AgentEvent(
                    type="message",
                    content={"role": "assistant", "content": "partial"},
                )
                raise RuntimeError("boom")

            async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
                return None

        with pytest.raises(AgentError, match="boom"):
            _consume(_make_agent(_BoomExecutor()), {"task": "hi", "instance_id": "i"})


class TestRunExecutorActivity:
    """The public workflow-activity wrapper around _consume_executor."""

    def test_run_executor_returns_final_message(self):
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                )
            ]
        )
        agent = _make_agent(executor)

        result = agent.run_executor(
            Mock(),
            {
                "task": "hi",
                "instance_id": "inst-1",
                "session_id": "inst-1",
                "stream_context": None,
            },
        )

        assert result == {"role": "assistant", "content": "done"}

    def test_run_executor_without_executor_raises(self):
        mock_llm = Mock(spec=OpenAIChatClient)
        mock_llm.prompt_template = None
        mock_llm.provider = "mock"
        mock_llm.api = "mock"
        mock_llm.model = "mock"
        mock_llm.component_name = None
        mock_llm.base_url = None
        mock_llm.azure_endpoint = None
        mock_llm.azure_deployment = None
        mock_llm.__class__.__name__ = "MockLLM"
        agent = DurableAgent(
            name="NoExec",
            role="NoExec",
            goal="NoExec",
            llm=mock_llm,
            pubsub=AgentPubSubConfig(pubsub_name="p", agent_topic="NoExec"),
            state=AgentStateConfig(store=StateStoreService(store_name="s")),
            registry=AgentRegistryConfig(store=StateStoreService(store_name="r")),
        )
        with pytest.raises(AgentError, match="without an AgentExecutorBase"):
            agent.run_executor(Mock(), {"task": "x", "instance_id": "i"})


class TestAgentWorkflowExecutorBranch:
    """Asserts the producer side: agent_workflow → run_executor activity input."""

    def _drive_workflow(self, agent: DurableAgent, message: Dict[str, Any]):
        """Pump the workflow generator and capture every call_activity payload."""
        from dapr.ext.workflow import DaprWorkflowContext

        ctx = DaprWorkflowContext()
        ctx.instance_id = "wf-inst-42"
        ctx.is_replaying = False

        captured: List[Dict[str, Any]] = []

        def track(activity, **kwargs):
            captured.append(
                {
                    "name": _activity_method_name(activity),
                    "raw_name": activity,
                    "input": kwargs.get("input"),
                    "retry_policy": kwargs.get("retry_policy"),
                }
            )
            if captured[-1]["name"] == "run_executor":
                return {"role": "assistant", "content": "done"}
            return None

        ctx.call_activity = Mock(side_effect=track)
        ctx.current_utc_datetime = Mock()
        ctx.current_utc_datetime.isoformat = Mock(
            return_value=getattr(self, "_started", "2026-04-27T00:00:00")
        )

        gen = agent.agent_workflow(ctx, message)
        result = None
        try:
            while True:
                result = gen.send(result)
        except StopIteration as exc:
            result = exc.value

        return captured, result

    def test_payload_omits_time_and_context_by_default(self):
        """Plain message ⇒ run_executor input has neither context nor time."""
        executor = _ScriptedExecutor([])  # never actually consumed; activity is mocked
        agent = _make_agent(executor)

        captured, _ = self._drive_workflow(agent, {"task": "hi"})

        run_calls = [c for c in captured if c["name"] == "run_executor"]
        assert len(run_calls) == 1
        payload = run_calls[0]["input"]
        assert "time" not in payload
        assert "context" not in payload
        assert payload["task"] == "hi"
        assert payload["instance_id"] == "wf-inst-42"
        assert run_calls[0]["retry_policy"] is agent._retry_policy

    def test_session_id_is_per_execution(self):
        """Without a caller session_id, replays and retries of one execution share
        an id (a retry resumes the saved session); a later execution reusing the
        instance id gets a new one."""
        agent = _make_agent(_ScriptedExecutor([]))

        def session_id(started: str) -> str:
            self._started = started
            captured, _ = self._drive_workflow(agent, {"task": "hi"})
            return next(c["input"] for c in captured if c["name"] == "run_executor")[
                "session_id"
            ]

        first = session_id("2026-04-27T00:00:00")
        assert first and first == session_id("2026-04-27T00:00:00")
        assert session_id("2026-05-01T09:30:00") != first

    def test_calls_executor_activity_by_registered_name(self):
        """The call site must use the agent-scoped name ``register_activities`` uses.

        Dapr resolves activities by registered name; a bare ``run_executor``
        reference does not match ``dapr.agents.<agent>.run_executor``.
        """
        agent = _make_agent(_ScriptedExecutor([]))

        captured, _ = self._drive_workflow(agent, {"task": "hi"})

        run_call = next(c for c in captured if c["name"] == "run_executor")
        assert run_call["raw_name"] == agent._activity_name(agent.run_executor)

    def test_payload_passes_caller_session_id_and_context(self):
        """Message with session_id/context ⇒ they flow into run_executor input."""
        executor = _ScriptedExecutor([])
        agent = _make_agent(executor)

        captured, _ = self._drive_workflow(
            agent,
            {
                "task": "hi",
                "session_id": "resume-7",
                "context": {"mcp_servers": ["primary"]},
            },
        )

        payload = next(c["input"] for c in captured if c["name"] == "run_executor")
        assert payload["session_id"] == "resume-7"
        assert payload["context"] == {"mcp_servers": ["primary"]}
        assert "time" not in payload
