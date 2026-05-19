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

"""Integration tests for :class:`DurableAgent`'s executor branch."""

import asyncio
import os
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import dapr.ext.workflow as wf

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from tests.agents.durableagent.test_durable_agent import _activity_method_name
from dapr_agents.agents.executors import (
    AgentEvent,
    AgentExecutorBase,
    EchoAgentExecutor,
)
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types import AgentError
from dapr_agents.types.tools import ToolExecutionStatus


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    """Neutralize Dapr workflow runtime + retry policy for unit tests."""
    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    class MockRetryPolicy:
        def __init__(
            self,
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=1),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout: Optional[timedelta] = None,
        ):
            self.max_number_of_attempts = max_number_of_attempts
            self.first_retry_interval = first_retry_interval
            self.max_retry_interval = max_retry_interval
            self.backoff_coefficient = backoff_coefficient
            self.retry_timeout = retry_timeout

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)
    yield mock_runtime


class MockDaprClient:
    """Minimal DaprClient stand-in that supports context-manager usage."""

    def __init__(self, http_timeout_seconds=10):
        self.get_state = MagicMock(return_value=Mock(data=None, json=lambda: {}))
        self.save_state = MagicMock()
        self.delete_state = MagicMock()
        self.query_state = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def get_metadata(self):
        response = MagicMock()
        response.registered_components = []
        response.application_id = "test-app-id"
        return response


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Mock DaprClient import sites so construction works offline."""
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    mock_client = MockDaprClient()
    mock_client.get_state.return_value = Mock(data=None)
    monkeypatch.setattr("dapr.clients.DaprClient", lambda *a, **kw: mock_client)
    monkeypatch.setattr(
        "dapr_agents.agents.base.DaprClient", lambda *a, **kw: mock_client
    )
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.statestore.DaprClient",
        lambda *a, **kw: mock_client,
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


def _make_agent(executor: AgentExecutorBase) -> DurableAgent:
    return DurableAgent(
        name="ExecAgent",
        role="Test Executor Agent",
        goal="Exercise the executor branch",
        executor=executor,
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
            agent_topic="ExecAgent",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        registry=AgentRegistryConfig(
            store=StateStoreService(store_name="testregistry")
        ),
        execution=AgentExecutionConfig(max_iterations=1),
    )


class _ScriptedExecutor(AgentExecutorBase):
    """Yields a pre-recorded script of events; captures ``run`` invocations."""

    def __init__(self, script: List[AgentEvent]) -> None:
        self._script = script
        self.calls: List[Dict[str, Any]] = []

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append(
            {"prompt": prompt, "session_id": session_id, "context": context}
        )
        for event in self._script:
            yield event

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return None


class TestDurableAgentConstruction:
    """Covers how the executor kwarg is wired through AgentBase."""

    def test_construct_with_executor_only(self):
        executor = EchoAgentExecutor()
        agent = _make_agent(executor)
        assert agent.executor is executor
        assert agent.llm is None

    def test_cannot_pass_both_llm_and_executor(self):
        mock_llm = Mock(spec=OpenAIChatClient)
        mock_llm.prompt_template = None
        executor = EchoAgentExecutor()
        with pytest.raises(ValueError, match="either `llm` or `executor`"):
            DurableAgent(
                name="Bad",
                role="Bad",
                goal="Bad",
                llm=mock_llm,
                executor=executor,
                pubsub=AgentPubSubConfig(pubsub_name="p", agent_topic="Bad"),
                state=AgentStateConfig(store=StateStoreService(store_name="s")),
                registry=AgentRegistryConfig(store=StateStoreService(store_name="r")),
            )


class TestConsumeExecutor:
    """Unit-tests the async consumer that the run_executor activity drives."""

    def _prime_entry(self, agent: DurableAgent) -> AgentWorkflowEntry:
        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry
        return entry

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
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            result = asyncio.run(
                agent._consume_executor(
                    {
                        "task": "hi",
                        "instance_id": "inst-1",
                        "session_id": "s",
                        "source": "test",
                    }
                )
            )

        assert result == {"role": "assistant", "content": "reply"}
        assert executor.calls == [{"prompt": "hi", "session_id": "s", "context": None}]
        # User + assistant message present in the workflow entry.
        roles = [getattr(m, "role", None) for m in entry.messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_session_event_triggers_save(self):
        executor = _ScriptedExecutor(
            [
                AgentEvent(type="session", content={}, session_id="s"),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                    session_id="s",
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state") as save_state,
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor(
                    {
                        "task": "hi",
                        "instance_id": "inst-1",
                        "session_id": "s",
                    }
                )
            )

        # At least two save_state calls: one on `session`, one terminal.
        assert save_state.call_count >= 2

    def test_tool_call_and_result_update_history(self):
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="tool_call",
                    content={
                        "id": "t1",
                        "name": "search",
                        "arguments": {"q": "x"},
                    },
                ),
                AgentEvent(
                    type="tool_result",
                    content={"tool_call_id": "t1", "result": "ok"},
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        assert len(entry.tool_history) == 1
        record = entry.tool_history[0]
        assert record.tool_call_id == "t1"
        assert record.tool_name == "search"
        assert record.status == ToolExecutionStatus.COMPLETED
        assert record.execution_result == "ok"

    def test_tool_result_after_session_checkpoint_is_persisted(self):
        """Session refresh must not orphan in-flight tool-call records.

        Simulates the real Dapr roundtrip: save_state snapshots the entry,
        get_state returns a freshly-validated copy (new Python objects).
        Without the tool_records rebuild, the tool_result update lands on
        stale references and the final persisted state stays ``RUNNING``.
        """
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="tool_call",
                    content={
                        "id": "t1",
                        "name": "search",
                        "arguments": {"q": "x"},
                    },
                ),
                AgentEvent(type="session", content={}),
                AgentEvent(
                    type="tool_result",
                    content={"tool_call_id": "t1", "result": "ok"},
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        saved: List[AgentWorkflowEntry] = []

        def _save(instance_id, entry=None):
            if entry is not None:
                saved.append(entry.model_copy(deep=True))

        def _get_state(wid):
            if not saved:
                return entry
            return saved[-1].model_copy(deep=True)

        with (
            patch.object(agent, "save_state", side_effect=_save),
            patch.object(agent._infra, "get_state", side_effect=_get_state),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        assert saved, "expected at least one save_state call"
        final = saved[-1]
        assert len(final.tool_history) == 1
        record = final.tool_history[0]
        assert record.tool_call_id == "t1"
        assert record.status == ToolExecutionStatus.COMPLETED
        assert record.execution_result == "ok"

    def test_error_event_raises_agent_error(self):
        executor = _ScriptedExecutor([AgentEvent(type="error", content="boom")])
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
            pytest.raises(AgentError, match="boom"),
        ):
            asyncio.run(
                agent._consume_executor({"task": "hi", "instance_id": "inst-1"})
            )

    def test_missing_complete_event_raises(self):
        executor = _ScriptedExecutor(
            [AgentEvent(type="message", content={"role": "assistant", "content": "x"})]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
            pytest.raises(AgentError, match="without a 'complete' event"),
        ):
            asyncio.run(
                agent._consume_executor({"task": "hi", "instance_id": "inst-1"})
            )

    def test_text_delta_is_not_persisted(self):
        executor = _ScriptedExecutor(
            [
                AgentEvent(type="text_delta", content="par"),
                AgentEvent(type="text_delta", content="tial"),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "partial"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        # No delta-sized artifacts bled into entry.messages.
        assistant_contents = [
            getattr(m, "content", None)
            for m in entry.messages
            if getattr(m, "role", None) == "assistant"
        ]
        assert "par" not in assistant_contents
        assert "tial" not in assistant_contents

    def test_caller_session_id_passes_through(self):
        """Caller-supplied session_id must reach executor.run unchanged."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                )
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor(
                    {
                        "task": "hi",
                        "instance_id": "inst-1",
                        "session_id": "sess-resume",
                    }
                )
            )

        assert executor.calls[-1]["session_id"] == "sess-resume"
        assert entry.session_id == "sess-resume"

    def test_omitted_session_id_yields_none_to_executor(self):
        """No payload session_id ⇒ executor.run gets None so it can auto-assign."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                )
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": "hi", "instance_id": "inst-1"})
            )

        assert executor.calls[-1]["session_id"] is None

    def test_retry_resumes_session_from_entry(self):
        """Retry-safe resumption: payload has no session_id but a prior attempt
        already checkpointed ``entry.session_id``. The activity must reuse that
        id instead of letting the executor mint a new one on every retry."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                )
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)
        # Simulate state left behind by a prior attempt that progressed past
        # the executor's first `session` checkpoint before failing.
        entry.session_id = "prior-sess-abc"

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": "retry", "instance_id": "inst-1"})
            )

        assert executor.calls[-1]["session_id"] == "prior-sess-abc"
        assert entry.session_id == "prior-sess-abc"

    def test_event_session_id_updates_entry(self):
        """Executor-assigned session_id from events must land on entry.session_id."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="message",
                    content={"role": "assistant", "content": "hi"},
                    session_id="exec-assigned-123",
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "hi"},
                    session_id="exec-assigned-123",
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": "hi", "instance_id": "inst-1"})
            )

        assert entry.session_id == "exec-assigned-123"

    def test_caller_context_threads_to_executor(self):
        """Caller-supplied context dict must reach executor.run unchanged."""
        ctx_payload = {"mcp_servers": ["primary"], "scopes": ["read"]}
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                )
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor(
                    {
                        "task": "hi",
                        "instance_id": "inst-1",
                        "context": ctx_payload,
                    }
                )
            )

        assert executor.calls[-1]["context"] == ctx_payload

    def test_tool_result_accepts_tool_use_id_alias(self):
        """Anthropic-style executors emit ``tool_use_id``; both keys must work."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="tool_call",
                    content={
                        "id": "t1",
                        "name": "search",
                        "arguments": {"q": "x"},
                    },
                ),
                AgentEvent(
                    type="tool_result",
                    content={"tool_use_id": "t1", "result": "ok"},
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        assert len(entry.tool_history) == 1
        record = entry.tool_history[0]
        assert record.tool_call_id == "t1"
        assert record.status == ToolExecutionStatus.COMPLETED
        assert record.execution_result == "ok"

    def test_tool_call_without_id_is_skipped(self):
        """Missing executor-provided id ⇒ warn and skip; do not invent one."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="tool_call",
                    content={"name": "search", "arguments": {"q": "x"}},
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        assert entry.tool_history == []

    def test_tool_result_without_id_is_skipped(self):
        """Missing tool_call_id/tool_use_id ⇒ warn and skip; do not invent one."""
        executor = _ScriptedExecutor(
            [
                AgentEvent(
                    type="tool_result",
                    content={"name": "search", "result": "ok"},
                ),
                AgentEvent(
                    type="complete",
                    content={"role": "assistant", "content": "done"},
                ),
            ]
        )
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            asyncio.run(
                agent._consume_executor({"task": None, "instance_id": "inst-1"})
            )

        assert entry.tool_history == []

    def test_state_persisted_in_finally_on_executor_exception(self):
        """A generic exception mid-stream must still flush accumulated state once."""

        class _BoomExecutor(AgentExecutorBase):
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            async def run(
                self,
                prompt: str,
                *,
                session_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
            ) -> AsyncIterator[AgentEvent]:
                self.calls.append({"prompt": prompt})
                yield AgentEvent(
                    type="message",
                    content={"role": "assistant", "content": "partial"},
                )
                raise RuntimeError("boom")

            async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
                return None

        executor = _BoomExecutor()
        agent = _make_agent(executor)
        entry = self._prime_entry(agent)

        with (
            patch.object(agent, "save_state") as save_state,
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
            pytest.raises(AgentError, match="boom"),
        ):
            asyncio.run(
                agent._consume_executor({"task": "hi", "instance_id": "inst-1"})
            )

        assert save_state.call_count == 1
        # The partial assistant message accumulated before the crash should be
        # in the entry that was flushed.
        assistant_contents = [
            getattr(m, "content", None)
            for m in entry.messages
            if getattr(m, "role", None) == "assistant"
        ]
        assert "partial" in assistant_contents


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
        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        with (
            patch.object(agent, "save_state"),
            patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        ):
            result = agent.run_executor(
                Mock(),
                {"task": "hi", "instance_id": "inst-1", "session_id": "inst-1"},
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
                    "input": kwargs.get("input"),
                    "retry_policy": kwargs.get("retry_policy"),
                }
            )
            if captured[-1]["name"] == "run_executor":
                return {"role": "assistant", "content": "done"}
            return None

        ctx.call_activity = Mock(side_effect=track)
        ctx.current_utc_datetime = Mock()
        ctx.current_utc_datetime.isoformat = Mock(return_value="2026-04-27T00:00:00")

        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[],
            tool_history=[],
        )
        agent._infra._state_model = entry

        with patch.object(agent._infra, "get_state", side_effect=lambda wid: entry):
            gen = agent.agent_workflow(ctx, message)
            result = None
            try:
                while True:
                    result = gen.send(result)
            except StopIteration as exc:
                result = exc.value

        return captured, result

    def test_payload_omits_time_session_and_context_by_default(self):
        """Plain message ⇒ run_executor input has neither session_id, context, nor time."""
        executor = _ScriptedExecutor([])  # never actually consumed; activity is mocked
        agent = _make_agent(executor)

        captured, _ = self._drive_workflow(agent, {"task": "hi"})

        run_calls = [c for c in captured if c["name"] == "run_executor"]
        assert len(run_calls) == 1
        payload = run_calls[0]["input"]
        assert "time" not in payload
        assert "session_id" not in payload
        assert "context" not in payload
        assert payload["task"] == "hi"
        assert payload["instance_id"] == "wf-inst-42"
        assert run_calls[0]["retry_policy"] is agent._retry_policy

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
