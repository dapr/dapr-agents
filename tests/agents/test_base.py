import pytest
import asyncio
import signal
from unittest.mock import Mock, patch

from dapr_agents.agents.base import AgentBase
from dapr_agents.memory import ConversationListMemory
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.prompt import ChatPromptTemplate
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import MessageContent, MessagePlaceHolder
from .mocks.llm_client import MockLLMClient
from .mocks.vectorstore import MockVectorStore


class TestAgentBase(AgentBase):
    """Concrete implementation of AgentBase for testing."""

    def run(self, input_data):
        """Implementation of abstract method for testing."""
        return f"Processed: {input_data}"


class TestAgentBaseClass:
    """Test cases for AgentBase class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return MockLLMClient()

    @pytest.fixture
    def basic_agent(self, mock_llm_client):
        """Create a basic test agent."""
        return TestAgentBase(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            instructions=["Test instruction 1", "Test instruction 2"],
            llm=mock_llm_client,
        )

    @pytest.fixture
    def minimal_agent(self, mock_llm_client):
        """Create a minimal test agent with only required fields."""
        return TestAgentBase(llm=mock_llm_client)

    @pytest.fixture
    def agent_with_system_prompt(self, mock_llm_client):
        """Create an agent with a custom system prompt."""
        return TestAgentBase(
            name="CustomAgent",
            system_prompt="You are a custom assistant. Help users with their questions.",
            llm=mock_llm_client,
        )

    @pytest.fixture
    def agent_with_tools(self, mock_llm_client):
        """Create an agent with tools."""
        mock_tool = Mock(spec=AgentTool)
        mock_tool.name = "test_tool"
        return TestAgentBase(name="ToolAgent", tools=[mock_tool], llm=mock_llm_client)

    @pytest.fixture
    def agent_with_vector_store(self, mock_llm_client):
        """Create an agent with vector store."""
        mock_vector_store = MockVectorStore()
        return TestAgentBase(
            name="VectorAgent", vector_store=mock_vector_store, llm=mock_llm_client
        )

    def test_agent_creation_with_all_fields(self, basic_agent):
        """Test agent creation with all fields specified."""
        assert basic_agent.name == "TestAgent"
        assert basic_agent.role == "Test Role"
        assert basic_agent.goal == "Test Goal"
        assert basic_agent.instructions == ["Test instruction 1", "Test instruction 2"]
        assert basic_agent.max_iterations == 10
        assert basic_agent.template_format == "jinja2"
        assert isinstance(basic_agent.memory, ConversationListMemory)
        assert basic_agent.llm is not None

    def test_agent_creation_with_minimal_fields(self, minimal_agent):
        """Test agent creation with minimal fields."""
        # Accept both None, 'Assistant', and 'Dapr Agent' for name
        assert minimal_agent.name in (None, "Assistant", "Dapr Agent")
        assert minimal_agent.role == "Assistant"
        assert minimal_agent.goal == "Help humans"
        assert minimal_agent.instructions is None
        # The system_prompt is automatically generated, so it won't be None
        assert minimal_agent.system_prompt is not None
        assert "Today's date is:" in minimal_agent.system_prompt

    def test_name_set_from_role_when_not_provided(self, mock_llm_client):
        """Test that name is set from role when not provided."""
        agent = TestAgentBase(role="Weather Expert", llm=mock_llm_client)
        assert agent.name == "Weather Expert"

    def test_name_not_overwritten_when_provided(self, mock_llm_client):
        """Test that name is not overwritten when explicitly provided."""
        agent = TestAgentBase(
            name="CustomName", role="Weather Expert", llm=mock_llm_client
        )
        assert agent.name == "CustomName"

    def test_agent_with_custom_system_prompt(self, agent_with_system_prompt):
        """Test agent with custom system prompt."""
        assert (
            agent_with_system_prompt.system_prompt
            == "You are a custom assistant. Help users with their questions."
        )
        assert agent_with_system_prompt.prompt_template is not None

    def test_agent_with_tools(self, agent_with_tools):
        """Test agent with tools."""
        assert len(agent_with_tools.tools) == 1
        assert agent_with_tools.tools[0].name == "test_tool"
        assert agent_with_tools.tool_executor is not None

    def test_agent_with_vector_store(self, agent_with_vector_store):
        """Test agent with vector store."""
        assert agent_with_vector_store.vector_store is not None

    def test_prompt_template_construction(self, basic_agent):
        """Test that prompt template is properly constructed."""
        assert basic_agent.prompt_template is not None
        assert isinstance(basic_agent.prompt_template, ChatPromptTemplate)
        # After pre-filling, only chat_history should remain in input_variables
        assert "chat_history" in basic_agent.prompt_template.input_variables

    def test_system_prompt_construction(self, basic_agent):
        """Test system prompt construction."""
        system_prompt = basic_agent.construct_system_prompt()
        assert "Today's date is:" in system_prompt
        assert "Your name is {{name}}." in system_prompt
        assert "Your role is {{role}}." in system_prompt
        assert "{{goal}}." in system_prompt
        assert "{{instructions}}" in system_prompt

    def test_system_prompt_without_instructions(self, mock_llm_client):
        """Test system prompt construction without instructions."""
        agent = TestAgentBase(
            name="TestAgent", role="Test Role", goal="Test Goal", llm=mock_llm_client
        )
        system_prompt = agent.construct_system_prompt()
        assert "{{instructions}}" not in system_prompt

    def test_prompt_template_construction_with_system_prompt(
        self, agent_with_system_prompt
    ):
        """Test prompt template construction with custom system prompt."""
        template = agent_with_system_prompt.construct_prompt_template()
        assert isinstance(template, ChatPromptTemplate)
        assert len(template.messages) == 2
        assert template.messages[0][0] == "system"
        assert isinstance(template.messages[1], MessagePlaceHolder)

    def test_construct_messages_with_string_input(self, basic_agent):
        """Test message construction with string input."""
        messages = basic_agent.construct_messages("Hello, how are you?")
        assert len(messages) > 0
        # Find the user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assert len(user_messages) == 2
        assert user_messages[-1]["content"] == "Hello, how are you?"

    def test_construct_messages_with_dict_input(self, basic_agent):
        """Test message construction with dictionary input."""
        # Use variables that are actually in the template
        input_data = {"chat_history": []}
        messages = basic_agent.construct_messages(input_data)
        assert len(messages) > 0

    def test_construct_messages_with_invalid_input(self, basic_agent):
        """Test message construction with invalid input."""
        with pytest.raises(
            ValueError, match="Input data must be either a string or dictionary"
        ):
            basic_agent.construct_messages(123)

    def test_get_last_message_empty_memory(self, basic_agent):
        """Test getting last message from empty memory."""
        assert basic_agent.get_last_message() is None

    def test_get_last_message_with_memory(self, basic_agent):
        """Test getting last message from memory with content."""
        # Create a mock message
        mock_message = Mock(spec=MessageContent)
        # Use patch.object to mock the method on the instance
        with patch.object(
            ConversationListMemory, "get_messages", return_value=[mock_message]
        ):
            result = basic_agent.get_last_message()
            assert result == mock_message

    def test_get_last_user_message(self, basic_agent):
        """Test getting last user message from message list."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "  User message with spaces  "},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "Last user message"},
        ]

        result = basic_agent.get_last_user_message(messages)
        assert result["role"] == "user"
        assert result["content"] == "Last user message"  # Should be trimmed

    def test_get_last_user_message_no_user_messages(self, basic_agent):
        """Test getting last user message when no user messages exist."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "assistant", "content": "Assistant response"},
        ]

        result = basic_agent.get_last_user_message(messages)
        assert result is None

    def test_reset_memory(self, basic_agent):
        """Test memory reset."""
        with patch.object(type(basic_agent.memory), "reset_memory") as mock_reset:
            basic_agent.reset_memory()
            mock_reset.assert_called_once()

    def test_pre_fill_prompt_template(self, basic_agent):
        """Test pre-filling prompt template with variables."""
        # Store original template for comparison
        original_template = basic_agent.prompt_template

        # Pre-fill with a variable
        basic_agent.pre_fill_prompt_template(custom_var="test_value")

        # Verify the template was updated
        assert basic_agent.prompt_template != original_template

        # Verify the pre-filled variable is set
        assert "custom_var" in basic_agent.prompt_template.pre_filled_variables
        assert (
            basic_agent.prompt_template.pre_filled_variables["custom_var"]
            == "test_value"
        )

        # Verify the template can still be formatted
        formatted = basic_agent.prompt_template.format_prompt()
        assert formatted is not None

    def test_pre_fill_prompt_template_without_template(self, mock_llm_client):
        """Test pre-filling prompt template when template is not initialized."""
        agent = TestAgentBase(llm=mock_llm_client)
        agent.prompt_template = None

        with pytest.raises(
            ValueError,
            match="Prompt template must be initialized before pre-filling variables",
        ):
            agent.pre_fill_prompt_template(custom_var="test_value")

    def test_chat_history_with_vector_memory_and_task(self):
        """Test chat history retrieval with vector memory and task."""
        from tests.agents.mocks.vectorstore import MockVectorStore
        from tests.agents.mocks.memory import DummyVectorMemory

        mock_vector_store = MockVectorStore()
        mock_llm = MockLLMClient()
        memory = DummyVectorMemory(mock_vector_store)
        agent = TestAgentBase(memory=memory, llm=mock_llm)

        # Access chat_history as a property
        result = agent.chat_history
        assert isinstance(result, list)
        assert isinstance(result[0], Mock)

    def test_chat_history_with_regular_memory(self, mock_llm_client):
        """Test chat history retrieval with regular memory."""
        memory = ConversationListMemory()
        agent = TestAgentBase(memory=memory, llm=mock_llm_client)

        with patch.object(
            ConversationListMemory,
            "get_messages",
            return_value=[Mock(spec=MessageContent)],
        ):
            result = agent.chat_history
            assert isinstance(result, list)
            assert isinstance(result[0], Mock)

    def test_prefill_agent_attributes_missing_fields_raises(self, mock_llm_client):
        """Test pre-filling agent attributes raises ValueError if fields are missing in the template."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "Just a system message"),
                MessagePlaceHolder(variable_name="chat_history"),
            ]
        )
        agent = TestAgentBase(
            name="TestAgent",
            role="TestRole",
            goal="TestGoal",
            instructions=["Do this", "Do that"],
            llm=mock_llm_client,
            prompt_template=prompt_template,
        )
        with pytest.raises(
            ValueError, match="The following agent attributes were explicitly set"
        ):
            agent.prefill_agent_attributes()

    def test_validate_llm_openai_without_api_key(self, monkeypatch):
        """Test validation fails when OpenAI is used without API key."""
        import openai
        from openai import OpenAI

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Temporarily restore the real OpenAI client for this test
        monkeypatch.setattr("openai.OpenAI", OpenAI)

        with pytest.raises(
            openai.OpenAIError, match="api_key client option must be set"
        ):
            TestAgentBase(llm=OpenAIChatClient())

    def test_validate_memory_failure(self, mock_llm_client):
        """Test validation fails when memory initialization fails."""
        with patch(
            "dapr_agents.memory.ConversationListMemory.__new__",
            side_effect=Exception("Memory error"),
        ):
            with pytest.raises(Exception, match="Memory error"):
                TestAgentBase(llm=mock_llm_client)

    def test_signal_handler_setup(self, basic_agent):
        """Test that signal handlers are set up."""
        assert hasattr(basic_agent, "_shutdown_event")
        assert isinstance(basic_agent._shutdown_event, asyncio.Event)

    def test_signal_handler(self, basic_agent):
        """Test signal handler functionality."""
        with patch("builtins.print") as mock_print:
            basic_agent._signal_handler(signal.SIGINT, None)
            mock_print.assert_called_once()
            assert basic_agent._shutdown_event.is_set()

    def test_conflicting_prompt_templates(self, mock_llm_client):
        """Test error when both agent and LLM have prompt templates."""
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = ChatPromptTemplate.from_messages(
            [("system", "test")]
        )
        mock_prompt_template = ChatPromptTemplate.from_messages([("system", "test2")])

        with pytest.raises(ValueError, match="Conflicting prompt templates"):
            TestAgentBase(llm=mock_llm, prompt_template=mock_prompt_template)

    def test_agent_with_custom_prompt_template(self, mock_llm_client):
        """Test agent with custom prompt template."""
        mock_prompt_template = ChatPromptTemplate.from_messages([("system", "test")])
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = None
        agent = TestAgentBase(llm=mock_llm, prompt_template=mock_prompt_template)
        assert agent.prompt_template is not None
        assert agent.llm.prompt_template is not None
        assert agent.prompt_template.messages == agent.llm.prompt_template.messages

    def test_agent_with_llm_prompt_template(self):
        """Test agent with LLM prompt template."""
        mock_prompt_template = ChatPromptTemplate.from_messages([("system", "test")])
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = mock_prompt_template
        agent = TestAgentBase(llm=mock_llm)
        assert agent.prompt_template is not None
        assert agent.llm.prompt_template is not None
        assert agent.prompt_template.messages == agent.llm.prompt_template.messages

    def test_run_method_implementation(self, basic_agent):
        """Test that the concrete run method works."""
        result = basic_agent.run("test input")
        assert result == "Processed: test input"

    def test_text_formatter_property(self, basic_agent):
        """Test text formatter property."""
        formatter = basic_agent.text_formatter
        assert formatter is not None

    def test_tool_executor_property(self, basic_agent):
        """Test tool executor property."""
        executor = basic_agent.tool_executor
        assert executor is not None

    def test_model_fields_set_detection(self, mock_llm_client):
        """Test that model_fields_set properly detects user-set attributes."""
        agent = TestAgentBase(
            name="TestName",  # User set
            role="TestRole",  # User set
            goal="TestGoal",  # User set
            llm=mock_llm_client,
        )

        # These should be in model_fields_set
        assert "name" in agent.model_fields_set
        assert "role" in agent.model_fields_set
        assert "goal" in agent.model_fields_set

    def test_template_format_validation(self, mock_llm_client):
        """Test template format validation."""
        agent = TestAgentBase(template_format="f-string", llm=mock_llm_client)
        assert agent.template_format == "f-string"

        agent = TestAgentBase(template_format="jinja2", llm=mock_llm_client)
        assert agent.template_format == "jinja2"

    def test_max_iterations_default(self, minimal_agent):
        """Test default max iterations."""
        assert minimal_agent.max_iterations == 10

    def test_max_iterations_custom(self, mock_llm_client):
        """Test custom max iterations."""
        agent = TestAgentBase(max_iterations=5, llm=mock_llm_client)
        assert agent.max_iterations == 5
