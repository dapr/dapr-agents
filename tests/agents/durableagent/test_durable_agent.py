# TODO(@Sicoyle): this test file is a bit of a mess, and needs to be refactored when we clean the remaining classes up.
# Right now we have to do a bunch of patching at the class-level instead of patching at the instance-level.
# In future, we should do dependency injection instead of patching at the class-level to make it easier to test.
# This applies to all areas in this file where we have with patch.object()...
import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from dapr_agents.agents.durableagent.agent import DurableAgent
from dapr_agents.agents.durableagent.schemas import (
    AgentTaskResponse,
    BroadcastMessage,
)
from dapr_agents.agents.durableagent.state import (
    AssistantWorkflowState,
)
from dapr_agents.memory import ConversationListMemory
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.tool.base import AgentTool
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.tool.executor import AgentToolExecutor


# We need this otherwise these tests all fail since they require Dapr to be available.
@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    from dapr_agents.workflow import agentic
    from dapr_agents.workflow import base
    from unittest.mock import Mock

    # Mock the Dapr availability check to always return True
    monkeypatch.setattr(
        agentic.AgenticWorkflow, "_is_dapr_available", lambda self: True
    )

    # Mock the WorkflowApp initialization to prevent DaprClient creation which does an internal check for Dapr availability.
    def mock_workflow_app_post_init(self, __context: Any) -> None:
        self.wf_runtime = Mock()
        self.wf_runtime_is_running = False
        self.wf_client = Mock()
        self.client = Mock()
        self.tasks = {}
        self.workflows = {}

        try:
            super(base.WorkflowApp, self).model_post_init(__context)
        except AttributeError:
            # If parent doesn't have model_post_init, that's fine
            pass

    monkeypatch.setattr(
        base.WorkflowApp, "model_post_init", mock_workflow_app_post_init
    )

    # Patch AssistantWorkflowState class to add __getitem__ and setdefault
    # TODO(@Sicoyle): in future, the AssistantWorkflowState class should be a Pydantic model,
    # and so we can not have to patch the class to add __getitem__ and setdefault,
    # because the data will not be a dictionary, but a Pydantic model with proper accessors.
    def _getitem(self, key):
        return getattr(self, key)

    def _setdefault(self, key, default):
        if hasattr(self, key):
            return getattr(self, key)
        setattr(self, key, default)
        return default

    AssistantWorkflowState.__getitem__ = _getitem
    AssistantWorkflowState.setdefault = _setdefault

    def mock_agentic_post_init(self, __context: Any) -> None:
        self._text_formatter = Mock()
        self._dapr_client = Mock()
        self._state_store_client = Mock()
        self._agent_metadata = {
            "name": getattr(self, "name", "TestAgent"),
            "role": getattr(self, "role", "Test Role"),
            "goal": getattr(self, "goal", "Test Goal"),
            "instructions": getattr(self, "instructions", []),
            "topic_name": getattr(
                self, "agent_topic_name", getattr(self, "name", "TestAgent")
            ),
            "pubsub_name": getattr(self, "message_bus_name", "testpubsub"),
            "orchestrator": False,
        }
        self._workflow_name = "ToolCallingWorkflow"
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._subscriptions = {}
        self._topic_handlers = {}

        if not hasattr(self, "state") or self.state is None:
            self.state = AssistantWorkflowState()

        # Call the WorkflowApp model_post_init which we have mocked above.
        super(agentic.AgenticWorkflow, self).model_post_init(__context)

    monkeypatch.setattr(
        agentic.AgenticWorkflow, "model_post_init", mock_agentic_post_init
    )

    # No-op for testing
    def mock_register_agentic_system(self):
        pass

    monkeypatch.setattr(
        agentic.AgenticWorkflow, "register_agentic_system", mock_register_agentic_system
    )

    yield


class TestDurableAgent:
    """Test cases for the DurableAgent class."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for testing."""
        os.environ["OPENAI_API_KEY"] = "test-api-key"
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
        context.call_activity = AsyncMock()
        context.wait_for_external_event = AsyncMock()
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
            memory=ConversationListMemory(),
            max_iterations=5,
            state_store_name="teststatestore",
            message_bus_name="testpubsub",
            agents_registry_store_name="testregistry",
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
            memory=ConversationListMemory(),
            tools=[mock_tool],
            max_iterations=5,
            state_store_name="teststatestore",
            message_bus_name="testpubsub",
            agents_registry_store_name="testregistry",
        )

    def test_durable_agent_initialization(self, mock_llm):
        """Test durable agent initialization with basic parameters."""
        agent = DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            instructions=["Be helpful"],
            llm=mock_llm,
            state_store_name="teststatestore",
            message_bus_name="testpubsub",
            agents_registry_store_name="testregistry",
        )

        assert agent.name == "TestDurableAgent"
        assert agent.role == "Test Durable Assistant"
        assert agent.goal == "Help with testing"
        assert agent.instructions == ["Be helpful"]
        assert agent.max_iterations == 10  # default value
        assert agent.tool_history == []
        assert agent.state_store_name == "teststatestore"
        assert agent.message_bus_name == "testpubsub"
        assert agent.agent_topic_name == "TestDurableAgent"
        assert agent.state is not None
        assert isinstance(agent.state, AssistantWorkflowState)

    def test_durable_agent_initialization_with_custom_topic(self, mock_llm):
        """Test durable agent initialization with custom topic name."""
        agent = DurableAgent(
            name="TestDurableAgent",
            role="Test Durable Assistant",
            goal="Help with testing",
            llm=mock_llm,
            agent_topic_name="custom-topic",
            state_store_name="teststatestore",
            message_bus_name="testpubsub",
            agents_registry_store_name="testregistry",
        )

        assert agent.agent_topic_name == "custom-topic"

    def test_durable_agent_initialization_name_from_role(self, mock_llm):
        """Test durable agent initialization with name derived from role."""
        agent = DurableAgent(
            role="Test Durable Assistant",
            goal="Help with testing",
            llm=mock_llm,
            state_store_name="teststatestore",
            message_bus_name="testpubsub",
            agents_registry_store_name="testregistry",
        )

        assert agent.name == "Test Durable Assistant"
        assert agent.agent_topic_name == "Test Durable Assistant"

    def test_durable_agent_metadata(self, basic_durable_agent):
        """Test durable agent metadata creation."""
        metadata = basic_durable_agent.agent_metadata

        assert metadata is not None
        assert metadata["name"] == "TestDurableAgent"
        assert metadata["role"] == "Test Durable Assistant"
        assert metadata["goal"] == "Help with testing"
        assert metadata["topic_name"] == "TestDurableAgent"
        assert metadata["pubsub_name"] == "testpubsub"
        assert metadata["orchestrator"] is False

    @pytest.mark.asyncio
    async def test_run_method(self, basic_durable_agent):
        """Test the run method returns appropriate message."""
        result = await basic_durable_agent.run("test input")

        assert "DurableAgent" in result
        assert "workflow service" in result
        assert "asynchronously" in result

    @pytest.mark.asyncio
    async def test_tool_calling_workflow_initialization(
        self, basic_durable_agent, mock_workflow_context
    ):
        """Test workflow initialization on first iteration."""
        message = {
            "task": "Test task",
            "iteration": 0,
            "workflow_instance_id": "parent-instance-123",
        }

        mock_workflow_context.instance_id = "test-instance-123"
        mock_workflow_context.call_activity.side_effect = [
            {"content": "Test response"},
            {"message": "Test response"},
            "stop",
        ]

        # Manually insert a mock instance to ensure the state is populated for the assertion
        from dapr_agents.agents.durableagent.state import AssistantWorkflowEntry

        basic_durable_agent.state.instances[
            "test-instance-123"
        ] = AssistantWorkflowEntry(
            input="Test task",
            source=None,
            source_workflow_instance_id="parent-instance-123",
        )

        workflow_gen = basic_durable_agent.tool_calling_workflow(
            mock_workflow_context, message
        )
        try:
            next(workflow_gen)
        except StopIteration:
            pass

        assert "test-instance-123" in basic_durable_agent.state.instances
        instance_data = basic_durable_agent.state.instances["test-instance-123"]
        assert instance_data.input == "Test task"
        assert instance_data.source is None
        assert instance_data.source_workflow_instance_id == "parent-instance-123"

    @pytest.mark.asyncio
    async def test_generate_response_activity(self, basic_durable_agent):
        """Test the generate_response activity."""
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "tool_calls": [],
                        "finish_reason": "stop",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        basic_durable_agent.llm.generate = Mock(return_value=mock_response)

        from dapr_agents.agents.durableagent.state import AssistantWorkflowEntry

        instance_id = "test-instance-123"
        workflow_entry = AssistantWorkflowEntry(
            input="Test task", source="test_source", source_workflow_instance_id=None
        )
        basic_durable_agent.state.instances = {instance_id: workflow_entry}

        result = await basic_durable_agent.generate_response(instance_id, "Test task")

        assert result == mock_response.model_dump.return_value
        basic_durable_agent.llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_message_activity(self, basic_durable_agent):
        """Test the get_response_message activity."""
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "tool_calls": [],
                        "finish_reason": "stop",
                    }
                }
            ]
        }

        result = basic_durable_agent.get_response_message(response)

        assert result["content"] == "Test response"
        assert result["tool_calls"] == []
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_get_finish_reason_activity(self, basic_durable_agent):
        """Test the get_finish_reason activity."""
        response = {"choices": [{"finish_reason": "stop"}]}

        result = basic_durable_agent.get_finish_reason(response)

        assert result == "stop"

    @pytest.mark.asyncio
    async def test_get_tool_calls_activity_with_tools(self, durable_agent_with_tools):
        """Test the get_tool_calls activity when tools are present."""
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"arg1": "value1"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }

        result = durable_agent_with_tools.get_tool_calls(response)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_get_tool_calls_activity_no_tools(self, basic_durable_agent):
        """Test the get_tool_calls activity when no tools are present."""
        response = {"tool_calls": []}

        result = basic_durable_agent.get_tool_calls(response)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_activity_success(self, durable_agent_with_tools):
        """Test successful tool execution in activity."""
        tool_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"arg1": "value1"}'},
        }
        from dapr_agents.agents.durableagent.state import AssistantWorkflowEntry

        instance_id = "test-instance-123"
        workflow_entry = AssistantWorkflowEntry(
            input="Test task", source="test_source", source_workflow_instance_id=None
        )
        durable_agent_with_tools.state.instances = {instance_id: workflow_entry}

        with patch.object(
            AgentToolExecutor,
            "run_tool",
            new_callable=AsyncMock,
            return_value="test_result",
        ) as mock_run_tool:
            await durable_agent_with_tools.execute_tool(instance_id, tool_call)
            instance_data = durable_agent_with_tools.state.instances[instance_id]
            assert len(instance_data.tool_history) == 1
            tool_entry = instance_data.tool_history[0]
            assert tool_entry.tool_call_id == "call_123"
            assert tool_entry.function_name == "test_tool"
            assert tool_entry.content == "test_result"
            mock_run_tool.assert_called_once_with("test_tool", arg1="value1")

    @pytest.mark.asyncio
    async def test_execute_tool_activity_failure(self, durable_agent_with_tools):
        """Test tool execution failure in activity."""
        tool_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"arg1": "value1"}'},
        }

        with patch.object(
            type(durable_agent_with_tools.tool_executor),
            "run_tool",
            side_effect=Exception("Tool failed"),
        ):
            with pytest.raises(Exception, match="Tool failed"):
                await durable_agent_with_tools.execute_tool(
                    "test-instance-123", tool_call
                )

    @pytest.mark.asyncio
    async def test_broadcast_message_to_agents_activity(self, basic_durable_agent):
        """Test broadcasting message to agents activity."""
        message = {
            "type": "broadcast",
            "content": "Test broadcast message",
            "sender": "TestDurableAgent",
        }

        with patch.object(
            type(basic_durable_agent), "broadcast_message"
        ) as mock_broadcast:
            await basic_durable_agent.broadcast_message_to_agents(message)
            mock_broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_response_back_activity(self, basic_durable_agent):
        """Test sending response back to target agent activity."""
        response = {"content": "Test response"}
        target_agent = "TargetAgent"
        target_instance_id = "target-instance-123"

        with patch.object(
            type(basic_durable_agent), "send_message_to_agent"
        ) as mock_send:
            await basic_durable_agent.send_response_back(
                response, target_agent, target_instance_id
            )
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_finish_workflow_activity(self, basic_durable_agent):
        """Test finishing workflow activity."""
        message = {"content": "Final response"}

        with patch.object(
            type(basic_durable_agent), "update_workflow_state"
        ) as mock_update:
            await basic_durable_agent.finish_workflow("test-instance-123", message)
            # The finish_workflow calls update_workflow_state twice, so check both calls
            assert mock_update.call_count == 2
            mock_update.assert_any_call(
                instance_id="test-instance-123", message=message
            )
            mock_update.assert_any_call(
                instance_id="test-instance-123", final_output="Final response"
            )

    @pytest.mark.asyncio
    async def test_update_workflow_state(self, basic_durable_agent):
        """Test updating workflow state."""
        instance_id = "test-instance-123"
        message = {"content": "Test message", "role": "user"}
        tool_message = {
            "tool_call_id": "call_123",
            "function_name": "test_tool",
            "content": "tool_result",
        }
        final_output = "Final output"

        from dapr_agents.agents.durableagent.state import AssistantWorkflowEntry

        workflow_entry = AssistantWorkflowEntry(
            input="Test task", source="test_source", source_workflow_instance_id=None
        )
        basic_durable_agent.state.instances = {instance_id: workflow_entry}

        await basic_durable_agent.update_workflow_state(
            instance_id, message, tool_message, final_output
        )

        instance_data = basic_durable_agent.state.instances[instance_id]
        assert len(instance_data.messages) == 1
        assert len(instance_data.tool_history) == 1
        assert instance_data.output == "Final output"

    @pytest.mark.asyncio
    async def test_broadcast_message(self, basic_durable_agent):
        """Test broadcasting message."""
        broadcast_msg = BroadcastMessage(
            content="Test broadcast",
            role="assistant",
            type="broadcast",
            sender="TestDurableAgent",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        await basic_durable_agent.broadcast_message(broadcast_msg)

    @pytest.mark.asyncio
    async def test_send_message_to_agent(self, basic_durable_agent):
        """Test sending message to specific agent."""
        task_response = AgentTaskResponse(
            content="Test task",
            role="assistant",
            task="Test task",
            agent_name="TargetAgent",
            workflow_instance_id="target-instance-123",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        basic_durable_agent.send_message_to_agent("TargetAgent", task_response)

    def test_register_agentic_system(self, basic_durable_agent):
        """Test registering agentic system."""
        # TODO(@Sicoyle): fix this to add assertions.
        basic_durable_agent.register_agentic_system()

    @pytest.mark.asyncio
    async def test_process_broadcast_message(self, basic_durable_agent):
        """Test processing broadcast message."""
        broadcast_msg = BroadcastMessage(
            content="Test broadcast",
            role="assistant",
            type="broadcast",
            sender="OtherAgent",
        )

        # This needs refactoring / better implementation on this test since the actual implementation would depend on the pubsub msg broker.
        await basic_durable_agent.process_broadcast_message(broadcast_msg)

    def test_durable_agent_properties(self, basic_durable_agent):
        """Test durable agent properties."""
        assert basic_durable_agent.tool_executor is not None
        assert basic_durable_agent.text_formatter is not None
        assert basic_durable_agent.state is not None

    def test_durable_agent_workflow_name(self, basic_durable_agent):
        """Test that the workflow name is set correctly."""
        assert basic_durable_agent._workflow_name == "ToolCallingWorkflow"

    def test_durable_agent_state_initialization(self, basic_durable_agent):
        """Test that the agent state is properly initialized."""
        assert isinstance(basic_durable_agent.state, AssistantWorkflowState)
        assert hasattr(basic_durable_agent.state, "instances")
        assert basic_durable_agent.state.instances == {}
