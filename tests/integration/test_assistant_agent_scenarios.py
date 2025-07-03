"""
AssistantAgent Integration Tests with Dapr Workflows

This test suite validates AssistantAgent workflow functionality including:
1. Basic workflow execution and state management
2. Tool calling within workflows
3. Multi-iteration workflows with memory
4. Error handling and recovery
5. Multi-agent communication (if infrastructure available)

Note: AssistantAgent requires Dapr Workflows runtime + Redis state store + pub/sub
"""

import pytest
from typing import Dict, Any
from dapr_agents.workflow.agents.assistant import AssistantAgent
from dapr_agents.tool import tool
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather conditions as a string.
    """
    return f"The weather in {location} is sunny and 75°F"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate, e.g. '2 + 2'

    Returns:
        Result of the calculation as a string.
    """
    try:
        # Simple evaluation for testing - in production use safer evaluation
        result = eval(expression.replace("^", "**"))
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@pytest.mark.integration
class TestAssistantAgentScenarios:
    """Test AssistantAgent workflow scenarios with Dapr."""

    @pytest.fixture
    def assistant_agent_basic(self):
        """Create a basic AssistantAgent for testing."""
        return AssistantAgent(
            name="TestAssistant",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful and accurate", "Use tools when appropriate"],
            llm=DaprChatClient(),
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="test_workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="test_agents_registry",
        )

    @pytest.fixture
    def assistant_agent_with_tools(self):
        """Create an AssistantAgent with tools for testing."""
        return AssistantAgent(
            name="ToolAssistant",
            role="Tool-Using Assistant",
            goal="Help with weather and calculations",
            instructions=[
                "Use weather tool for weather questions",
                "Use calculator for math",
            ],
            tools=[get_weather, calculate],
            llm=DaprChatClient(),
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="test_tool_workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="test_agents_registry",
        )

    @pytest.fixture
    def assistant_agent_with_memory(self):
        """Create an AssistantAgent with persistent memory."""
        return AssistantAgent(
            name="MemoryAssistant",
            role="Assistant with Memory",
            goal="Remember conversation history",
            instructions=[
                "Remember what users tell you",
                "Reference previous conversations",
            ],
            tools=[get_weather],
            llm=DaprChatClient(),
            memory=ConversationDaprStateMemory(
                store_name="conversationstore", session_id="test-session-123"
            ),
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="test_memory_workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="test_agents_registry",
        )

    def test_assistant_agent_initialization(self, assistant_agent_basic):
        """Test that AssistantAgent initializes correctly."""
        agent = assistant_agent_basic

        # Verify basic properties
        assert agent.name == "TestAssistant"
        assert agent.role == "Test Assistant"
        assert agent._workflow_name == "ToolCallingWorkflow"
        assert agent.state is not None
        assert isinstance(agent.state, dict)

        # Verify workflow configuration
        assert agent.message_bus_name == "messagepubsub"
        assert agent.state_store_name == "workflowstatestore"
        assert agent.agents_registry_store_name == "registrystatestore"

        print("✅ AssistantAgent initialized correctly")

    def test_assistant_agent_tool_configuration(self, assistant_agent_with_tools):
        """Test that AssistantAgent configures tools correctly."""
        agent = assistant_agent_with_tools

        # Verify tool configuration
        assert len(agent.tools) == 2
        assert agent.tool_choice == "auto"  # Should be auto when tools provided

        # Verify tool names
        tool_names = [tool.name for tool in agent.tools]
        assert "get_weather" in tool_names
        assert "calculate" in tool_names

        print("✅ AssistantAgent tools configured correctly")

    @pytest.mark.integration
    async def test_assistant_agent_workflow_runtime_setup(self, assistant_agent_basic):
        """Test that AssistantAgent sets up workflow runtime correctly."""
        agent = assistant_agent_basic

        # Verify workflow runtime components
        assert agent.wf_runtime is not None
        assert agent.wf_client is not None
        assert agent.client is not None

        # Verify workflow registration
        assert "ToolCallingWorkflow" in agent.workflows

        # Verify task registration
        expected_tasks = [
            "generate_response",
            "get_response_message",
            "get_finish_reason",
            "get_tool_calls",
            "execute_tool",
            "finish_workflow",
        ]

        for task_name in expected_tasks:
            assert task_name in agent.tasks

        print("✅ AssistantAgent workflow runtime setup correctly")

    @pytest.mark.integration
    async def test_assistant_agent_basic_workflow_execution(
        self, assistant_agent_basic
    ):
        """Test basic workflow execution without tools."""
        agent = assistant_agent_basic

        # Start the workflow runtime
        agent.start_runtime()

        try:
            # Prepare workflow input
            workflow_input = {"task": "Hello, can you help me?", "iteration": 0}

            # Run workflow
            instance_id = agent.run_workflow("ToolCallingWorkflow", workflow_input)
            assert instance_id is not None

            # Monitor workflow completion
            state = await agent.monitor_workflow_state(instance_id)
            assert state is not None

            # Verify workflow completed successfully
            assert state.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Verify workflow state was updated
            assert instance_id in agent.state.get("instances", {})
            workflow_entry = agent.state["instances"][instance_id]
            assert workflow_entry["input"] == "Hello, can you help me?"
            assert workflow_entry["output"] is not None

            print("✅ Basic AssistantAgent workflow executed successfully")

        finally:
            agent.stop_runtime()

    @pytest.mark.integration
    async def test_assistant_agent_tool_calling_workflow(
        self, assistant_agent_with_tools
    ):
        """Test workflow execution with tool calling."""
        agent = assistant_agent_with_tools

        # Start the workflow runtime
        agent.start_runtime()

        try:
            # Prepare workflow input that should trigger tool usage
            workflow_input = {
                "task": "What's the weather like in San Francisco?",
                "iteration": 0,
            }

            # Run workflow
            instance_id = agent.run_workflow("ToolCallingWorkflow", workflow_input)
            assert instance_id is not None

            # Monitor workflow completion
            state = await agent.monitor_workflow_state(instance_id)
            assert state is not None

            # Verify workflow completed successfully
            assert state.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Verify tool execution was recorded
            workflow_entry = agent.state["instances"][instance_id]
            assert len(workflow_entry.get("tool_history", [])) > 0

            # Verify tool was called correctly
            tool_execution = workflow_entry["tool_history"][0]
            assert tool_execution["function_name"] == "get_weather"
            assert "San Francisco" in tool_execution["function_args"]
            assert "sunny and 75°F" in tool_execution["content"]

            print("✅ AssistantAgent tool calling workflow executed successfully")

        finally:
            agent.stop_runtime()

    @pytest.mark.integration
    async def test_assistant_agent_multi_iteration_workflow(
        self, assistant_agent_with_tools
    ):
        """Test multi-iteration workflow with multiple tool calls."""
        agent = assistant_agent_with_tools

        # Start the workflow runtime
        agent.start_runtime()

        try:
            # Prepare workflow input that should trigger multiple iterations
            workflow_input = {
                "task": "What's the weather in New York and what's 15 + 25?",
                "iteration": 0,
            }

            # Run workflow
            instance_id = agent.run_workflow("ToolCallingWorkflow", workflow_input)
            assert instance_id is not None

            # Monitor workflow completion
            state = await agent.monitor_workflow_state(instance_id)
            assert state is not None

            # Verify workflow completed successfully
            assert state.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Verify multiple tools were executed
            workflow_entry = agent.state["instances"][instance_id]
            tool_history = workflow_entry.get("tool_history", [])
            assert len(tool_history) >= 2  # Should have both weather and calculation

            # Verify both tools were called
            function_names = [tool["function_name"] for tool in tool_history]
            assert "get_weather" in function_names
            assert "calculate" in function_names

            print("✅ AssistantAgent multi-iteration workflow executed successfully")

        finally:
            agent.stop_runtime()

    @pytest.mark.integration
    async def test_assistant_agent_memory_persistence(
        self, assistant_agent_with_memory
    ):
        """Test that AssistantAgent persists memory across workflow executions."""
        agent = assistant_agent_with_memory

        # Start the workflow runtime
        agent.start_runtime()

        try:
            # First workflow - establish context
            workflow_input_1 = {
                "task": "My name is Alice and I live in Seattle",
                "iteration": 0,
            }

            instance_id_1 = agent.run_workflow("ToolCallingWorkflow", workflow_input_1)
            state_1 = await agent.monitor_workflow_state(instance_id_1)
            assert state_1.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Second workflow - reference previous context
            workflow_input_2 = {
                "task": "What's the weather where I live?",
                "iteration": 0,
            }

            instance_id_2 = agent.run_workflow("ToolCallingWorkflow", workflow_input_2)
            state_2 = await agent.monitor_workflow_state(instance_id_2)
            assert state_2.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Verify second workflow used context from first
            workflow_entry_2 = agent.state["instances"][instance_id_2]
            tool_history_2 = workflow_entry_2.get("tool_history", [])

            # Should have called weather tool with Seattle
            weather_calls = [
                tool
                for tool in tool_history_2
                if tool["function_name"] == "get_weather"
            ]
            assert len(weather_calls) > 0
            assert "Seattle" in weather_calls[0]["function_args"]

            print("✅ AssistantAgent memory persistence working correctly")

        finally:
            agent.stop_runtime()

    @pytest.mark.integration
    async def test_assistant_agent_error_handling(self, assistant_agent_with_tools):
        """Test AssistantAgent error handling in workflows."""
        agent = assistant_agent_with_tools

        # Start the workflow runtime
        agent.start_runtime()

        try:
            # Prepare workflow input that should cause a calculation error
            workflow_input = {"task": "Calculate 1/0 please", "iteration": 0}

            # Run workflow
            instance_id = agent.run_workflow("ToolCallingWorkflow", workflow_input)
            assert instance_id is not None

            # Monitor workflow completion
            state = await agent.monitor_workflow_state(instance_id)
            assert state is not None

            # Workflow should still complete even with tool error
            assert state.runtime_status.name == "ORCHESTRATION_STATUS_COMPLETED"

            # Verify error was handled gracefully
            workflow_entry = agent.state["instances"][instance_id]
            tool_history = workflow_entry.get("tool_history", [])

            if tool_history:
                calc_tool = next(
                    (
                        tool
                        for tool in tool_history
                        if tool["function_name"] == "calculate"
                    ),
                    None,
                )
                if calc_tool:
                    assert "Error" in calc_tool["content"]

            print("✅ AssistantAgent error handling working correctly")

        finally:
            agent.stop_runtime()

    def test_assistant_agent_state_management(self, assistant_agent_basic):
        """Test AssistantAgent state management functionality."""
        agent = assistant_agent_basic

        # Test initial state
        assert agent.state == {}

        # Test state initialization for workflow instance
        instance_id = "test-instance-123"
        agent.state["instances"] = {}

        # Simulate workflow entry creation
        from dapr_agents.workflow.agents.assistant.state import AssistantWorkflowEntry

        workflow_entry = AssistantWorkflowEntry(
            input="Test input", source="test_source"
        )

        agent.state["instances"][instance_id] = workflow_entry.model_dump(mode="json")

        # Verify state structure
        assert instance_id in agent.state["instances"]
        entry = agent.state["instances"][instance_id]
        assert entry["input"] == "Test input"
        assert entry["source"] == "test_source"
        assert entry["messages"] == []
        assert entry["tool_history"] == []

        print("✅ AssistantAgent state management working correctly")

    def validate_workflow_response(self, response: Dict[str, Any]) -> None:
        """Validate the structure of workflow response."""
        assert isinstance(response, dict)
        # Add specific workflow response validation logic

    def validate_tool_execution(self, tool_execution: Dict[str, Any]) -> None:
        """Validate tool execution structure."""
        required_fields = ["tool_call_id", "function_name", "content"]
        for field in required_fields:
            assert field in tool_execution
            assert tool_execution[field] is not None
