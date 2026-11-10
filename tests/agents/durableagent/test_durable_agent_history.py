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

``test_durable_agent`` holds the old path's copy; at the contract release
the old copy goes and this module takes its name.
"""

import os
from unittest.mock import AsyncMock, Mock
import pytest
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentStateConfig,
    AgentRegistryConfig,
    AgentMemoryConfig,
    AgentExecutionConfig,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import AgentError
from tests.agents.durableagent.test_durable_agent import (  # noqa: F401
    _activity_method_name,
    patch_dapr_check,
    MockDaprClient,
)

pytestmark = pytest.mark.usefixtures("history_path_only")


class TestDurableAgent:
    """Test cases for the DurableAgent class."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock DaprClient to use our context manager supporting mock
        mock_client = MockDaprClient()
        mock_client.get_state.return_value = Mock(data=None)  # Default empty state

        # Patch both the client import locations
        monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
            lambda: mock_client,
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = Mock(spec=OpenAIChatClient)
        mock.generate = AsyncMock()
        mock.prompt_template = None
        # Set the class name to avoid OpenAI validation
        mock.__class__.__name__ = "MockLLMClient"
        mock.provider = "MockOpenAIProvider"
        mock.api = "MockOpenAIAPI"
        mock.model = "gpt-4o-mock"
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock(spec=AgentTool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.run = AsyncMock(return_value="test_result")
        tool._is_async = True
        return tool

    @pytest.fixture
    def mock_workflow_context(self):
        """Create a mock Dapr workflow context."""
        context = DaprWorkflowContext()
        context.instance_id = "test-instance-123"
        context.is_replaying = False
        context.call_activity = Mock()
        context.wait_for_external_event = Mock()
        context.set_custom_status = Mock()
        context.current_utc_datetime = Mock()
        context.current_utc_datetime.isoformat = Mock(
            return_value="2024-01-01T00:00:00.000000"
        )
        return context

    @pytest.fixture
    def basic_durable_agent(self, mock_llm):
        """Create a basic durable agent instance for testing."""
        return DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Test things"],
            llm=mock_llm,
            pubsub=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="TestDurableAgent",
            ),
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore",
                    workflow_instance_id="test_session",
                )
            ),
            execution=AgentExecutionConfig(max_iterations=5),
        )

    @pytest.fixture
    def durable_agent_with_tools(self, mock_llm, mock_tool):
        """Create a durable agent with tools for testing."""
        return DurableAgent(
            name="ToolDurableAgent",
            role="Tool Durable Assistant",
            goal="Execute tools",
            instructions=["Use tools when needed"],
            llm=mock_llm,
            pubsub=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="ToolDurableAgent",
            ),
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore",
                    workflow_instance_id="test_session",
                )
            ),
            tools=[mock_tool],
            execution=AgentExecutionConfig(max_iterations=5),
        )

    def test_first_activity_is_call_llm_with_the_task(
        self, basic_durable_agent, mock_workflow_context
    ):
        """The first call_llm carries the task as its only new message."""
        message = {"task": "Test task", "workflow_instance_id": "parent-instance-123"}
        mock_workflow_context.instance_id = "test-instance-123"

        workflow_gen = basic_durable_agent.agent_workflow(
            mock_workflow_context, message
        )
        next(workflow_gen)  # load_tools (the agent has a registry)
        workflow_gen.send([])

        activity, kwargs = mock_workflow_context.call_activity.call_args
        assert _activity_method_name(activity[0]) == "call_llm"
        assert kwargs["input"]["new_messages"] == [
            {"role": "user", "content": "Test task"}
        ]
        assert kwargs["input"]["history_len"] == 0
        assert kwargs["input"]["last_id"] is None

    def test_finalize_workflow_forgets_cached_conversations(self, basic_durable_agent):
        """finalize_workflow drops only the finished instance's cached conversation."""
        agent = basic_durable_agent
        agent._remember("wf-1", [{"role": "assistant", "content": "a", "id": "x"}])
        agent._remember("wf-2", [{"role": "assistant", "content": "b", "id": "z"}])

        agent.finalize_workflow(
            Mock(), {"instance_id": "wf-1", "final_message": {}, "stream_context": None}
        )

        assert "wf-1" not in agent._conversations
        assert "wf-2" in agent._conversations

    def test_recent_broadcasts(self, basic_durable_agent):
        """The last three team broadcasts are offered to the LLM as context."""
        basic_durable_agent.memory = Mock()
        basic_durable_agent.memory.get_messages.return_value = [
            {"role": "user", "content": f"b{i}", "createdAt": "t"} for i in range(5)
        ]

        assert basic_durable_agent._recent_broadcasts() == [
            {"role": "user", "content": "b2"},
            {"role": "user", "content": "b3"},
            {"role": "user", "content": "b4"},
        ]
        basic_durable_agent.memory.get_messages.assert_called_once_with("broadcast")

    def test_agent_workflow_applies_retry_policy(
        self, basic_durable_agent, mock_workflow_context
    ):
        """Test that agent_workflow applies retry policy to activity calls."""
        message = {
            "task": "Test task with retries",
            "workflow_instance_id": "parent-instance-123",
        }

        call_activity_calls = []

        def track_call_activity(activity, **kwargs):
            call_activity_calls.append(
                {
                    "activity": activity,
                    "input": kwargs.get("input"),
                    "retry_policy": kwargs.get("retry_policy"),
                }
            )

            activity_name = _activity_method_name(activity)

            if activity_name == "call_llm":
                return {
                    "content": "Test response",
                    "tool_calls": [
                        {
                            "id": "call_test_123",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": '{"arg": "value"}',
                            },
                        }
                    ],
                    "role": "assistant",
                }
            elif activity_name == "run_tool":
                return {
                    "tool_call_id": "call_test_123",
                    "content": "tool result",
                    "role": "tool",
                    "name": "test_tool",
                }
            elif activity_name in [
                "finalize_workflow",
            ]:
                return None

        mock_workflow_context.instance_id = "test-instance-123"
        mock_workflow_context.call_activity = Mock(side_effect=track_call_activity)

        # Run the workflow generator
        workflow_gen = basic_durable_agent.agent_workflow(
            mock_workflow_context, message
        )

        # Step through the generator, sending results back
        result = None
        try:
            while True:
                result = workflow_gen.send(result)
        except StopIteration as e:
            result = e.value

        # Verify that retry_policy was passed to critical activities
        assert len(call_activity_calls) >= 5, (
            f"Expected at least 3 activity calls, got {len(call_activity_calls)}"
        )

        # All activities should have retry_policy parameter
        for call in call_activity_calls:
            assert "retry_policy" in call, f"Missing retry_policy in call: {call}"
            assert call["retry_policy"] == basic_durable_agent._retry_policy, (
                f"Expected retry_policy {basic_durable_agent._retry_policy}, got {call['retry_policy']}"
            )

        # Verify the key activities were called
        # Activities are registered under agent-scoped names
        # (dapr.agents.<AgentName>.<method>); match on the trailing method
        # segment so the assertion doesn't break if the scoping format
        # changes later.
        activity_names = []
        for call in call_activity_calls:
            raw = getattr(call["activity"], "__name__", str(call["activity"]))
            activity_names.append(raw.rsplit(".", 1)[-1] if "." in raw else raw)
        assert "call_llm" in activity_names, f"Missing call_llm in {activity_names}"
        assert "run_tool" in activity_names, f"Missing run_tool in {activity_names}"
        assert "finalize_workflow" in activity_names, (
            f"Missing finalize_workflow in {activity_names}"
        )

    def test_agent_workflow_max_iterations_sets_custom_status(
        self, basic_durable_agent, mock_workflow_context, mock_tool
    ):
        """Test that set_custom_status('max_iterations_reached') is called when max iterations is reached."""
        message = {"task": "Test task"}

        # Set up agent with max_iterations=2 and register a tool
        basic_durable_agent.execution.max_iterations = 2
        basic_durable_agent.tool_executor.register_tool(mock_tool)

        # Track call count to ensure we always return tool calls for call_llm
        call_llm_count = 0

        # Mock call_activity to return responses with tool_calls to force iterations
        def mock_call_activity(activity, **kwargs):
            nonlocal call_llm_count
            activity_name = _activity_method_name(activity)

            if activity_name == "call_llm":
                call_llm_count += 1
                # Always return tool calls to force continuation (never return final response)
                return {
                    "content": "I'll use a tool",
                    "tool_calls": [
                        {
                            "id": f"call_{call_llm_count}",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                    "role": "assistant",
                }
            elif activity_name == "run_tool":
                return {
                    "tool_call_id": f"call_{call_llm_count}",
                    "content": "tool result",
                    "role": "tool",
                    "name": "test_tool",
                }
            elif activity_name == "load_tools":
                return []  # Empty list of registered tools
            elif activity_name in [
                "finalize_workflow",
                "broadcast_to_team",
                "return_response",
                "summarize",
            ]:
                return None
            return None

        mock_workflow_context.call_activity = Mock(side_effect=mock_call_activity)

        workflow_gen = basic_durable_agent.agent_workflow(
            mock_workflow_context, message
        )
        # Consume the generator until it completes
        result = None
        try:
            while True:
                result = workflow_gen.send(result)
        except StopIteration:
            pass

        # Verify set_custom_status was called with "max_iterations_reached"
        mock_workflow_context.set_custom_status.assert_called_once_with(
            "max_iterations_reached"
        )

    def test_agent_workflow_exception_raises_agent_error(
        self, basic_durable_agent, mock_workflow_context
    ):
        """Test that an exception in the workflow results in AgentError being raised."""
        message = {"task": "Test task"}

        # Mock call_activity to raise an exception
        test_exception = RuntimeError("Test workflow failure")

        def mock_call_activity(activity, **kwargs):
            activity_name = _activity_method_name(activity)

            if activity_name == "load_tools":
                return []  # Empty list of registered tools
            elif activity_name == "call_llm":
                # Raise exception on LLM call
                raise test_exception
            elif activity_name in [
                "broadcast_to_team",
                "return_response",
                "summarize",
                "finalize_workflow",
            ]:
                return None
            return None

        mock_workflow_context.call_activity = Mock(side_effect=mock_call_activity)

        workflow_gen = basic_durable_agent.agent_workflow(
            mock_workflow_context, message
        )
        # Consume the generator until it raises AgentError
        # The exception is caught internally, workflow continues to finalize,
        # then AgentError is raised at the end
        with pytest.raises(AgentError) as exc_info:
            try:
                while True:
                    next(workflow_gen)
            except StopIteration:
                # Should not reach here - AgentError should be raised first
                pass

        # Verify the exception message contains the original error and agent name
        assert "workflow failed" in str(exc_info.value).lower()
        assert "TestDurableAgent" in str(exc_info.value)
        # Verify it's chained to the original exception
        assert exc_info.value.__cause__ == test_exception

    @pytest.mark.parametrize(
        "memory_kwargs, expect_summarize",
        [
            ({}, True),
            ({"summarize_on_completion": True}, True),
            ({"summarize_on_completion": False}, False),
        ],
        ids=["default", "explicit-true", "opt-out"],
    )
    def test_agent_workflow_summarize_gated_by_memory_config(
        self, mock_llm, mock_workflow_context, memory_kwargs, expect_summarize
    ):
        """The end-of-run summarize activity runs unless the memory config opts out."""
        memory = AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="teststatestore",
                workflow_instance_id="test_session",
            ),
            **memory_kwargs,
        )
        agent = DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            llm=mock_llm,
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            memory=memory,
        )
        # Opting out of the summary must not disable memory itself.
        assert agent.memory is not None

        activity_names = []

        def mock_call_activity(activity, **kwargs):
            activity_name = _activity_method_name(activity)
            activity_names.append(activity_name)
            if activity_name == "call_llm":
                return {"role": "assistant", "content": "done"}
            if activity_name == "load_tools":
                return []
            return None

        mock_workflow_context.call_activity = Mock(side_effect=mock_call_activity)

        workflow_gen = agent.agent_workflow(
            mock_workflow_context, {"task": "Test task"}
        )
        result = None
        try:
            while True:
                result = workflow_gen.send(result)
        except StopIteration:
            pass

        assert "finalize_workflow" in activity_names
        assert ("summarize" in activity_names) is expect_summarize, activity_names


class TestCallLlmRequestShape:
    """call_llm must omit tools/tool_choice when starting a fresh structured-output
    turn (response_format set, last assistant has no tool_calls), and keep them
    otherwise. This eliminates the "model emits tool calls instead of JSON"
    failure mode without breaking mid-tool-loop message validity.
    """

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        os.environ["OPENAI_API_KEY"] = "test-api-key"
        mock_client = MockDaprClient()
        mock_client.get_state.return_value = Mock(data=None)
        monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
            lambda: mock_client,
        )
        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @pytest.fixture
    def mock_llm(self):
        from dapr_agents.types.message import (
            LLMChatResponse,
            LLMChatCandidate,
            AssistantMessage,
        )

        mock = Mock(spec=OpenAIChatClient)
        mock.prompt_template = None
        mock.__class__.__name__ = "MockLLMClient"
        mock.provider = "openai"
        mock.api = "MockOpenAIAPI"
        mock.model = "gpt-4o-mock"

        def _generate(**kwargs):
            return LLMChatResponse(
                results=[
                    LLMChatCandidate(
                        message=AssistantMessage(content="ok"),
                        finish_reason="stop",
                    )
                ]
            )

        mock.generate = Mock(side_effect=_generate)
        return mock

    @pytest.fixture
    def agent_with_tool(self, mock_llm):
        tool = self._make_tool()
        return DurableAgent(
            name="ShapeAgent",
            role="Test",
            goal="Test",
            instructions=["x"],
            llm=mock_llm,
            pubsub=AgentPubSubConfig(
                pubsub_name="testpubsub", agent_topic="ShapeAgent"
            ),
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore",
                    workflow_instance_id="test_session",
                )
            ),
            tools=[tool],
            execution=AgentExecutionConfig(max_iterations=5),
        )

    def _make_tool(self, name="test_tool"):
        tool = Mock(spec=AgentTool)
        tool.name = name
        tool.description = "A test tool"
        tool.run = AsyncMock(return_value="result")
        tool._is_async = True
        return tool

    def _make_agent(self, mock_llm, *, execution, tools=None):
        return DurableAgent(
            name="LateToolChoiceAgent",
            role="Test",
            goal="Test",
            instructions=["x"],
            llm=mock_llm,
            tools=tools,
            execution=execution,
        )

    def _stub_call_llm_helpers(self, agent, messages):
        """Patch internal helpers so call_llm runs with controlled messages."""
        agent._conversation = Mock(return_value=[])
        agent.prompting_helper = Mock()
        agent.prompting_helper.build_initial_messages = Mock(return_value=messages)
        agent.text_formatter = Mock()

    def _ctx(self):
        ctx = Mock(spec=DaprWorkflowContext)
        ctx.instance_id = "wf-shape-1"
        return ctx

    def _call_llm(self, agent):
        messages = [{"role": "user", "content": "hi"}]
        self._stub_call_llm_helpers(agent, messages)
        agent.call_llm(
            self._ctx(),
            payload={"instance_id": "wf-shape-1", "source": "direct", "history_len": 0},
        )
        return agent.llm.generate.call_args.kwargs

    def test_no_response_format_passes_tools_normally(self, agent_with_tool):
        messages = [{"role": "user", "content": "hi"}]
        self._stub_call_llm_helpers(agent_with_tool, messages)

        agent_with_tool.call_llm(
            self._ctx(),
            payload={"instance_id": "wf-shape-1", "source": "direct", "history_len": 0},
        )

        kwargs = agent_with_tool.llm.generate.call_args.kwargs
        assert "tools" in kwargs and kwargs["tools"], (
            "tools should be passed in non-structured turn"
        )
        assert "tool_choice" in kwargs
        assert "response_format" not in kwargs

    def test_response_format_without_prior_tool_calls_omits_tools(
        self, agent_with_tool
    ):
        # Last assistant message has no tool_calls → fresh structured turn.
        messages = [
            {"role": "user", "content": "summarize"},
            {"role": "assistant", "content": "ok"},
        ]
        self._stub_call_llm_helpers(agent_with_tool, messages)

        agent_with_tool.call_llm(
            self._ctx(),
            payload={
                "instance_id": "wf-shape-1",
                "source": "direct",
                "history_len": 0,
                "response_format": "IterablePlanStep",
            },
        )

        kwargs = agent_with_tool.llm.generate.call_args.kwargs
        assert "response_format" in kwargs, (
            "structured request must include response_format"
        )
        assert "tools" not in kwargs, (
            "tools must be omitted on fresh structured-output turn so the model "
            "can't legitimately respond with a tool call instead of JSON"
        )
        assert "tool_choice" not in kwargs

    def test_response_format_with_prior_tool_calls_keeps_tools(self, agent_with_tool):
        # Last assistant message DID emit tool_calls → mid-loop. Tools must
        # remain attached so tool_call_id references stay valid.
        messages = [
            {"role": "user", "content": "search and summarize"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "x", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        self._stub_call_llm_helpers(agent_with_tool, messages)

        agent_with_tool.call_llm(
            self._ctx(),
            payload={
                "instance_id": "wf-shape-1",
                "source": "direct",
                "history_len": 0,
                "response_format": "ProgressCheckOutput",
            },
        )

        kwargs = agent_with_tool.llm.generate.call_args.kwargs
        assert "response_format" in kwargs
        assert "tools" in kwargs and kwargs["tools"], (
            "tools must remain when the conversation is mid-tool-loop"
        )
        assert "tool_choice" in kwargs

    def test_structured_output_failure_wraps_with_diagnostic_context(
        self, agent_with_tool
    ):

        messages = [{"role": "user", "content": "plan"}]
        self._stub_call_llm_helpers(agent_with_tool, messages)
        agent_with_tool.llm.generate = Mock(
            side_effect=ValueError(
                "OpenAI API error (StructureError): No content found for JSON mode."
            )
        )

        with pytest.raises(AgentError) as exc_info:
            agent_with_tool.call_llm(
                self._ctx(),
                payload={
                    "instance_id": "wf-shape-1",
                    "source": "direct",
                    "history_len": 0,
                    "response_format": "IterablePlanStep",
                },
            )

        msg = str(exc_info.value)
        assert "structured-output call failed" in msg
        assert "schema='IterablePlanStep'" in msg
        assert "provider='openai'" in msg
        assert "model='gpt-4o-mock'" in msg
        assert "agent='ShapeAgent'" in msg
