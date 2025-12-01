import pytest
import asyncio
import os
from unittest.mock import Mock, patch
from dapr_agents.agents.standalone import Agent
from dapr_agents.agents.configs import AgentMemoryConfig, AgentExecutionConfig
from dapr_agents.types import (
    AgentError,
    AssistantMessage,
    LLMChatResponse,
    ToolExecutionRecord,
    UserMessage,
    ToolCall,
)
from dapr_agents.memory import ConversationListMemory
from dapr_agents.llm import OpenAIChatClient
from tests.agents.agent.testtools import echo_tool, error_tool


class TestAgent:
    """Test cases for the Agent class."""

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
        mock.generate = Mock()
        mock.prompt_template = None
        # Set the class name to avoid OpenAI validation
        mock.__class__.__name__ = "MockLLMClient"
        return mock

    @pytest.fixture
    def basic_agent(self, mock_llm):
        """Create a basic agent instance for testing."""
        return Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful", "Test things"],
            llm=mock_llm,
            memory=AgentMemoryConfig(store=ConversationListMemory()),
            execution=AgentExecutionConfig(max_iterations=5),
        )

    @pytest.fixture
    def agent_with_tools(self, mock_llm):
        """Create an agent with tools for testing."""
        return Agent(
            name="ToolAgent",
            role="Tool Assistant",
            goal="Execute tools",
            instructions=["Use tools when needed"],
            llm=mock_llm,
            memory=AgentMemoryConfig(store=ConversationListMemory()),
            tools=[echo_tool],
            execution=AgentExecutionConfig(max_iterations=5),
        )

    def test_agent_initialization(self, mock_llm):
        """Test agent initialization with basic parameters."""
        agent = Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            instructions=["Be helpful"],
            llm=mock_llm,
            tools=[echo_tool],
        )

        assert agent.name == "TestAgent"
        assert agent.prompting_helper.role == "Test Assistant"
        assert agent.prompting_helper.goal == "Help with testing"
        assert agent.prompting_helper.instructions == ["Be helpful"]
        assert agent.execution.max_iterations == 10  # default value
        assert agent.tool_history == []
        assert agent.execution.tool_choice == "auto"  # auto when tools are provided

    def test_agent_initialization_without_tools(self, mock_llm):
        """Test agent initialization without tools."""
        agent = Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm,
        )

        assert agent.execution.tool_choice is None  # no tools → no tool_choice

    def test_agent_initialization_with_custom_tool_choice(self, mock_llm):
        """Test agent initialization with custom tool choice."""
        agent = Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm,
            tools=[echo_tool],
            execution=AgentExecutionConfig(tool_choice="required"),
        )

        assert agent.execution.tool_choice == "required"

    @pytest.mark.asyncio
    async def test_run_with_shutdown_event(self, basic_agent):
        """Test agent run method with shutdown event."""
        basic_agent._shutdown_event.set()

        result = await basic_agent.run("test input")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_with_cancellation(self, basic_agent):
        """Test agent run method with cancellation."""
        with patch.object(
            basic_agent, "_run_agent", side_effect=asyncio.CancelledError
        ):
            result = await basic_agent.run("test input")
            assert result is None

    @pytest.mark.asyncio
    async def test_run_with_exception(self, basic_agent):
        """Test agent run method with exception."""
        with patch.object(
            basic_agent, "_run_agent", side_effect=Exception("Test error")
        ):
            with pytest.raises(Exception, match="Test error"):
                await basic_agent.run("test input")

    @pytest.mark.asyncio
    async def test_run_agent_basic(self, basic_agent):
        """Test basic agent run functionality."""
        mock_response = Mock(spec=LLMChatResponse)
        assistant_msg = AssistantMessage(content="Hello!")
        mock_response.get_message.return_value = assistant_msg
        basic_agent.llm.generate.return_value = mock_response

        result = await basic_agent._run_agent(
            input_data="Hello", instance_id="test-123"
        )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello!"
        basic_agent.llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_with_tool_calls(self, agent_with_tools):
        mock_function = Mock()
        mock_function.name = echo_tool.name
        mock_function.arguments = '{"arg1": "value1"}'
        mock_function.arguments_dict = {"arg1": "value1"}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.type = "function"
        tool_call.function = mock_function

        first_response = Mock(spec=LLMChatResponse)
        first_assistant = AssistantMessage(content="Using tool", tool_calls=[tool_call])
        first_response.get_message.return_value = first_assistant

        second_response = Mock(spec=LLMChatResponse)
        second_assistant = AssistantMessage(content="Final answer")
        second_response.get_message.return_value = second_assistant

        agent_with_tools.llm.generate.side_effect = [first_response, second_response]
        agent_with_tools.tools = [echo_tool]
        agent_with_tools.tool_executor = agent_with_tools.tool_executor.__class__(
            tools=[echo_tool]
        )

        result = await agent_with_tools._run_agent(
            input_data="Use the tool", instance_id="test-123"
        )
        assert isinstance(result, AssistantMessage)
        assert result.content == "Final answer"

    async def test_process_response_success(self, agent_with_tools):
        """Test successful tool execution."""
        mock_function = Mock()
        mock_function.name = echo_tool.name
        mock_function.arguments = '{"arg1": "value1"}'
        mock_function.arguments_dict = {"arg1": "value1"}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.function = mock_function
        agent_with_tools.tools = [echo_tool]
        agent_with_tools.tool_executor = agent_with_tools.tool_executor.__class__(
            tools=[echo_tool]
        )

        # Call the actual internal method that executes tool calls
        tool_messages = await agent_with_tools._execute_tool_calls(
            "test-instance", [tool_call]
        )

        assert len(agent_with_tools.tool_history) == 1
        tool_record = agent_with_tools.tool_history[0]
        assert tool_record.tool_call_id == "call_123"
        assert tool_record.tool_name == echo_tool.name
        assert tool_record.execution_result == "value1"

        # Verify the tool message was returned
        assert len(tool_messages) == 1
        assert tool_messages[0]["role"] == "tool"
        assert tool_messages[0]["name"] == echo_tool.name

    async def test_process_response_failure(self, agent_with_tools):
        mock_function = Mock()
        mock_function.name = error_tool.name
        mock_function.arguments = "{}"
        mock_function.arguments_dict = {}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.function = mock_function
        agent_with_tools.tools = [error_tool]
        agent_with_tools.tool_executor = agent_with_tools.tool_executor.__class__(
            tools=[error_tool]
        )

        # Call the actual internal method that executes tool calls
        tool_record = await agent_with_tools._execute_tool_calls(
            "test-instance", [tool_call]
        )

        # Verify the error result
        assert "isError=True" in tool_record[0]["content"]
        assert "Tool failed" in tool_record[0]["content"]

    async def test_conversation_max_reached(self, basic_agent):
        """Test that agent stops immediately when there are no tool calls."""
        mock_response = Mock(spec=LLMChatResponse)
        assistant_msg = AssistantMessage(content="Using tool", tool_calls=[])
        mock_response.get_message.return_value = assistant_msg
        basic_agent.llm.generate.return_value = mock_response

        # Call the actual internal conversation loop method
        initial_messages = [{"role": "user", "content": "Hello"}]
        result = await basic_agent._conversation_loop(
            instance_id="test-123", messages=initial_messages
        )

        # current logic sees no tools ===> returns on first iteration
        assert isinstance(result, AssistantMessage)
        assert result.content == "Using tool"
        assert basic_agent.llm.generate.call_count == 1

    async def test_conversation_with_llm_error(self, basic_agent):
        """Test handling of LLM errors during iterations."""
        basic_agent.llm.generate.side_effect = Exception("LLM error")

        # Call the actual internal conversation loop method
        initial_messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            AgentError, match="Failed during chat generation: LLM error"
        ):
            await basic_agent._conversation_loop(
                instance_id="test-123", messages=initial_messages
            )

    async def test_run_tool_success(self, agent_with_tools):
        """Test successful tool execution via _run_tool_call method."""
        agent_with_tools.tools = [echo_tool]
        agent_with_tools.tool_executor = agent_with_tools.tool_executor.__class__(
            tools=[echo_tool]
        )

        # Create a mock tool call
        mock_function = Mock()
        mock_function.name = echo_tool.name
        mock_function.arguments_dict = {"arg1": "value1"}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.function = mock_function

        # Call the actual internal method
        result = await agent_with_tools._run_tool_call("test-instance", tool_call)

        # Verify the result is a tool message dict
        assert result["role"] == "tool"
        assert result["name"] == echo_tool.name
        assert result["content"] == "value1"

    async def test_run_tool_failure(self, agent_with_tools):
        """Test tool execution failure via _run_tool_call method."""
        agent_with_tools.tools = [error_tool]
        agent_with_tools.tool_executor = agent_with_tools.tool_executor.__class__(
            tools=[error_tool]
        )

        # Create a mock tool call
        mock_function = Mock()
        mock_function.name = error_tool.name
        mock_function.arguments_dict = {}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.function = mock_function

        # Call the actual internal method
        result = await agent_with_tools._run_tool_call("test-instance", tool_call)

        # Verify the error result - content should contain error message
        assert "Tool failed" in result["content"]

    def test_agent_properties(self, basic_agent):
        """Test agent properties."""
        assert basic_agent.tool_executor is not None
        assert basic_agent.text_formatter is not None

    @pytest.mark.asyncio
    async def test_agent_with_memory_context(self, basic_agent):
        """Test agent using memory context when no input is provided."""
        basic_agent.memory.add_message(UserMessage(content="Previous message"))

        mock_response = Mock(spec=LLMChatResponse)
        assistant_msg = AssistantMessage(content="Response")
        mock_response.get_message.return_value = assistant_msg
        basic_agent.llm.generate.return_value = mock_response

        result = await basic_agent._run_agent(input_data=None, instance_id="test-123")

        assert isinstance(result, AssistantMessage)
        assert result.content == "Response"
        basic_agent.llm.generate.assert_called_once()

    def test_agent_tool_history_management(self, basic_agent):
        """Test tool history management."""
        tool_execution_record = ToolExecutionRecord(
            tool_call_id="call_123",
            tool_name="echo_tool",
            tool_args={"arg1": "value1"},
            execution_result="test_result",
        )
        basic_agent.tool_history.append(tool_execution_record)

        assert len(basic_agent.tool_history) == 1
        assert basic_agent.tool_history[0].tool_name == "echo_tool"

        basic_agent.tool_history.clear()
        assert len(basic_agent.tool_history) == 0
