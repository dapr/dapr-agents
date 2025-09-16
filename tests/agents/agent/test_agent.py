import pytest
import asyncio
import os
from unittest.mock import Mock, patch
from dapr_agents.agents.agent.agent import Agent
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
            memory=ConversationListMemory(),
            max_iterations=5,
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
            memory=ConversationListMemory(),
            tools=[echo_tool],
            max_iterations=5,
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
        assert agent.role == "Test Assistant"
        assert agent.goal == "Help with testing"
        assert agent.instructions == ["Be helpful"]
        assert agent.max_iterations == 10  # default value
        assert agent.tool_history == []
        assert agent.tool_choice == "auto"  # auto when tools are provided

    def test_agent_initialization_without_tools(self, mock_llm):
        """Test agent initialization without tools."""
        agent = Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm,
        )

        assert agent.tool_choice is None

    def test_agent_initialization_with_custom_tool_choice(self, mock_llm):
        """Test agent initialization with custom tool choice."""
        agent = Agent(
            name="TestAgent",
            role="Test Assistant",
            goal="Help with testing",
            llm=mock_llm,
            tool_choice="required",
        )

        assert agent.tool_choice == "required"

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

        result = await basic_agent._run_agent("Hello")

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
        tool_call.function = mock_function

        first_response = Mock(spec=LLMChatResponse)
        first_assistant = AssistantMessage(content="Using tool", tool_calls=[tool_call])
        first_response.get_message.return_value = first_assistant

        second_response = Mock(spec=LLMChatResponse)
        second_assistant = AssistantMessage(content="Final answer")
        second_response.get_message.return_value = second_assistant

        agent_with_tools.llm.generate.side_effect = [first_response, second_response]
        agent_with_tools.tools = [echo_tool]
        agent_with_tools._tool_executor = agent_with_tools._tool_executor.__class__(
            tools=[echo_tool]
        )

        result = await agent_with_tools._run_agent("Use the tool")
        assert isinstance(result, AssistantMessage)
        assert result.content == "Final answer"

    @pytest.mark.asyncio
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
        agent_with_tools._tool_executor = agent_with_tools._tool_executor.__class__(
            tools=[echo_tool]
        )
        await agent_with_tools.execute_tools([tool_call])
        assert len(agent_with_tools.tool_history) == 1
        tool_message = agent_with_tools.tool_history[0]
        assert tool_message.tool_call_id == "call_123"
        assert tool_message.tool_name == echo_tool.name
        assert tool_message.execution_result == "value1"

    @pytest.mark.asyncio
    async def test_process_response_failure(self, agent_with_tools):
        mock_function = Mock()
        mock_function.name = error_tool.name
        mock_function.arguments = "{}"
        mock_function.arguments_dict = {}
        tool_call = Mock(spec=ToolCall)
        tool_call.id = "call_123"
        tool_call.function = mock_function
        agent_with_tools.tools = [error_tool]
        agent_with_tools._tool_executor = agent_with_tools._tool_executor.__class__(
            tools=[error_tool]
        )
        with pytest.raises(
            AgentError, match=f"Error executing tool '{error_tool.name}': .*Tool failed"
        ):
            await agent_with_tools.execute_tools([tool_call])

    @pytest.mark.asyncio
    async def test_conversation_max_reached(self, basic_agent):
        """Test that agent stops immediately when there are no tool calls."""
        mock_response = Mock(spec=LLMChatResponse)
        assistant_msg = AssistantMessage(content="Using tool", tool_calls=[])
        mock_response.get_message.return_value = assistant_msg
        basic_agent.llm.generate.return_value = mock_response

        result = await basic_agent.conversation([])

        # current logic sees no tools ===> returns on first iteration
        assert isinstance(result, AssistantMessage)
        assert result.content == "Using tool"
        assert basic_agent.llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_conversation_with_llm_error(self, basic_agent):
        """Test handling of LLM errors during iterations."""
        basic_agent.llm.generate.side_effect = Exception("LLM error")

        with pytest.raises(
            AgentError, match="Failed during chat generation: LLM error"
        ):
            await basic_agent.conversation([])

    @pytest.mark.asyncio
    async def test_run_tool_success(self, agent_with_tools):
        """Test successful tool execution via run_tool method."""
        agent_with_tools.tools = [echo_tool]
        agent_with_tools._tool_executor = agent_with_tools._tool_executor.__class__(
            tools=[echo_tool]
        )
        result = await agent_with_tools.run_tool(echo_tool.name, arg1="value1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_run_tool_failure(self, agent_with_tools):
        """Test tool execution failure via run_tool method."""
        agent_with_tools.tools = [error_tool]
        agent_with_tools._tool_executor = agent_with_tools._tool_executor.__class__(
            tools=[error_tool]
        )
        with pytest.raises(
            AgentError, match=f"Failed to run tool '{error_tool.name}': .*Tool failed"
        ):
            await agent_with_tools.run_tool(error_tool.name)

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

        result = await basic_agent._run_agent(None)

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
