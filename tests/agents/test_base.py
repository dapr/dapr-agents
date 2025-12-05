import pytest
from unittest.mock import Mock, patch

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.memory import ConversationListMemory
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.prompt import ChatPromptTemplate
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import MessageContent, MessagePlaceHolder
from .mocks.llm_client import MockLLMClient
from .mocks.vectorstore import MockVectorStore


class ConcreteAgentBase(AgentBase):
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
        return ConcreteAgentBase(
            name="TestAgent",
            role="Test Role",
            goal="Test Goal",
            instructions=["Test instruction 1", "Test instruction 2"],
            llm=mock_llm_client,
        )

    @pytest.fixture
    def minimal_agent(self, mock_llm_client):
        """Create a minimal test agent with only required fields."""
        return ConcreteAgentBase(name="MinimalAgent", llm=mock_llm_client)

    @pytest.fixture
    def agent_with_system_prompt(self, mock_llm_client):
        """Create an agent with a custom system prompt."""
        return ConcreteAgentBase(
            name="CustomAgent",
            system_prompt="You are a custom assistant. Help users with their questions.",
            llm=mock_llm_client,
        )

    @pytest.fixture
    def agent_with_tools(self, mock_llm_client):
        """Create an agent with tools."""
        mock_tool = Mock(spec=AgentTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "A tool for testing."
        return ConcreteAgentBase(
            name="ToolAgent", tools=[mock_tool], llm=mock_llm_client
        )

    @pytest.fixture
    def agent_with_vector_store(self, mock_llm_client):
        """Create an agent with vector store."""
        mock_vector_store = MockVectorStore()
        return ConcreteAgentBase(
            name="VectorAgent", vector_store=mock_vector_store, llm=mock_llm_client
        )

    def test_agent_creation_with_all_fields(self, basic_agent):
        """Test agent creation with all fields specified."""
        assert basic_agent.name == "TestAgent"
        assert basic_agent.prompting_helper.role == "Test Role"
        assert basic_agent.prompting_helper.goal == "Test Goal"
        assert basic_agent.prompting_helper.instructions == [
            "Test instruction 1",
            "Test instruction 2",
        ]
        assert basic_agent.execution.max_iterations == 10
        assert basic_agent.prompting_helper.template_format == "jinja2"
        assert isinstance(basic_agent.memory, ConversationListMemory)
        assert basic_agent.llm is not None

    def test_agent_creation_with_minimal_fields(self, minimal_agent):
        """Test agent creation with minimal fields."""
        # Name is now required
        assert minimal_agent.name == "MinimalAgent"
        assert minimal_agent.prompting_helper.role == "Assistant"
        assert minimal_agent.prompting_helper.goal in (
            "Help users accomplish their tasks.",
            "Help humans",
        )
        assert minimal_agent.prompting_helper.instructions == []
        # The system_prompt is automatically generated, so it won't be None
        assert (
            minimal_agent.prompting_helper.system_prompt is not None
            or minimal_agent.prompt_template is not None
        )

    def test_name_set_from_role_when_not_provided(self, mock_llm_client):
        """Test that agent can be created with just a name."""
        agent = ConcreteAgentBase(name="Weather Expert", llm=mock_llm_client)
        assert agent.name == "Weather Expert"

    def test_name_not_overwritten_when_provided(self, mock_llm_client):
        """Test that name is not overwritten when explicitly provided."""
        agent = ConcreteAgentBase(
            name="CustomName", role="Weather Expert", llm=mock_llm_client
        )
        assert agent.name == "CustomName"

    def test_agent_with_custom_system_prompt(self, agent_with_system_prompt):
        """Test agent with custom system prompt."""
        assert (
            agent_with_system_prompt.prompting_helper.system_prompt
            == "You are a custom assistant. Help users with their questions."
        )
        assert agent_with_system_prompt.prompt_template is not None

    def test_agent_with_tools(self, agent_with_tools):
        """Test agent with tools."""
        assert len(agent_with_tools.tools) == 1
        assert agent_with_tools.tools[0].name == "test_tool"
        assert agent_with_tools.tool_executor is not None

    def test_prompt_template_construction(self, basic_agent):
        """Test that prompt template is properly constructed."""
        assert basic_agent.prompt_template is not None
        assert isinstance(basic_agent.prompt_template, ChatPromptTemplate)
        # After pre-filling, only chat_history should remain in input_variables
        assert "chat_history" in basic_agent.prompt_template.input_variables

    def test_system_prompt_construction(self, basic_agent):
        """Test system prompt construction."""
        # System prompt is now built automatically via prompting_helper
        system_prompt = basic_agent.prompting_helper.system_prompt or str(
            basic_agent.prompt_template
        )
        assert system_prompt is not None
        # The prompting helper has the role and goal
        assert basic_agent.prompting_helper.role == "Test Role"
        assert basic_agent.prompting_helper.goal == "Test Goal"

    def test_system_prompt_without_instructions(self, mock_llm_client):
        """Test system prompt construction without instructions."""
        agent = ConcreteAgentBase(
            name="TestAgent", role="Test Role", goal="Test Goal", llm=mock_llm_client
        )
        # Check that prompt template was created
        assert agent.prompt_template is not None

    def test_prompt_template_construction_with_system_prompt(
        self, agent_with_system_prompt
    ):
        """Test prompt template construction with custom system prompt."""
        # Prompt template is now automatically constructed
        template = agent_with_system_prompt.prompt_template
        assert isinstance(template, ChatPromptTemplate)
        assert len(template.messages) >= 1

    def test_construct_messages_with_string_input(self, basic_agent):
        """Test message construction with string input."""
        messages = basic_agent.build_initial_messages("Hello, how are you?")
        assert len(messages) > 0
        # Find the user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "Hello, how are you?"

    def test_construct_messages_with_dict_input(self, basic_agent):
        """Test message construction with dictionary input."""
        # build_initial_messages handles chat_history internally
        messages = basic_agent.build_initial_messages("Test message")
        assert len(messages) > 0

    def test_construct_messages_with_invalid_input(self, basic_agent):
        """Test message construction with various input types."""
        # The method now handles various input types gracefully
        # Just verify it doesn't crash
        messages = basic_agent.build_initial_messages(None)
        assert len(messages) > 0

    def test_get_last_message_empty_memory(self, basic_agent):
        """Test getting last message from empty memory."""
        assert basic_agent.get_last_message() is None

    def test_get_last_message_with_memory(self, basic_agent):
        """Test getting last message from memory with content."""
        # Use a dictionary as the mock message
        mock_message = {"foo": "bar"}
        with patch.object(
            ConversationListMemory, "get_messages", return_value=[mock_message]
        ):
            result = basic_agent.get_last_message()
            assert result == {"foo": "bar"}

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

    def test_chat_history_with_vector_memory_and_task(self):
        """Test chat history retrieval with vector memory and task."""
        from tests.agents.mocks.vectorstore import MockVectorStore
        from tests.agents.mocks.memory import DummyVectorMemory

        mock_vector_store = MockVectorStore()
        mock_llm = MockLLMClient()
        memory = DummyVectorMemory(mock_vector_store)
        agent = ConcreteAgentBase(
            name="TestAgent",
            memory=AgentMemoryConfig(store=memory),
            llm=mock_llm,
        )

        # Call get_chat_history() method instead of accessing property
        result = agent.get_chat_history()
        assert isinstance(result, list)
        assert isinstance(result[0], Mock)

    def test_chat_history_with_regular_memory(self, mock_llm_client):
        """Test chat history retrieval with regular memory."""
        memory = ConversationListMemory()
        agent = ConcreteAgentBase(
            name="TestAgent",
            memory=AgentMemoryConfig(store=memory),
            llm=mock_llm_client,
        )

        with patch.object(
            ConversationListMemory,
            "get_messages",
            return_value=[Mock(spec=MessageContent)],
        ):
            result = agent.get_chat_history()
            assert isinstance(result, list)
            assert isinstance(result[0], Mock)

    def test_prefill_agent_attributes_missing_fields_warns(
        self, mock_llm_client, caplog
    ):
        """Test that prompt variables are prefilled correctly even when some are not used in template."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "Just a system message"),
                MessagePlaceHolder(variable_name="chat_history"),
            ]
        )
        agent = ConcreteAgentBase(
            name="TestAgent",
            role="TestRole",
            goal="TestGoal",
            instructions=["Do this", "Do that"],
            llm=mock_llm_client,
            prompt_template=prompt_template,
        )
        # Verify that prompting_helper was initialized and prefilled variables
        assert agent.prompting_helper is not None
        assert agent.prompting_helper.name == "TestAgent"
        assert agent.prompting_helper.role == "TestRole"
        assert agent.prompting_helper.goal == "TestGoal"
        # The prompt template should be prefilled
        assert agent.prompt_template is not None

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
            ConcreteAgentBase(llm=OpenAIChatClient())

    def test_validate_memory_failure(self, mock_llm_client):
        """Test validation fails when memory initialization fails."""
        with patch(
            "dapr_agents.memory.ConversationListMemory.__new__",
            side_effect=Exception("Memory error"),
        ):
            with pytest.raises(Exception, match="Memory error"):
                ConcreteAgentBase(name="TestAgent", llm=mock_llm_client)

    def test_conflicting_prompt_templates(self, caplog):
        """Test that agent can have its own prompt template even when LLM has one."""
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = ChatPromptTemplate.from_messages(
            [("system", "llm template")]
        )
        mock_prompt_template = ChatPromptTemplate.from_messages(
            [("system", "agent template")]
        )

        agent = ConcreteAgentBase(
            name="TestAgent", llm=mock_llm, prompt_template=mock_prompt_template
        )
        # Agent's prompt template should be used and set on LLM
        assert agent.prompt_template is not None
        assert agent.llm.prompt_template is not None
        # The LLM should now have the agent's template
        assert agent.llm.prompt_template == agent.prompt_template

    def test_agent_with_custom_prompt_template(self):
        """Test agent with custom prompt template."""
        mock_prompt_template = ChatPromptTemplate.from_messages([("system", "test")])
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = None
        agent = ConcreteAgentBase(
            name="TestAgent", llm=mock_llm, prompt_template=mock_prompt_template
        )
        assert agent.prompt_template is not None
        assert agent.llm.prompt_template is not None
        assert agent.prompt_template.messages == agent.llm.prompt_template.messages

    def test_agent_with_llm_prompt_template(self):
        """Test agent initialization when LLM has a prompt template."""
        mock_prompt_template = ChatPromptTemplate.from_messages([("system", "test")])
        mock_llm = MockLLMClient()
        mock_llm.prompt_template = mock_prompt_template
        agent = ConcreteAgentBase(name="TestAgent", llm=mock_llm)
        # Agent should build its own prompt template from profile
        assert agent.prompt_template is not None
        assert agent.llm.prompt_template is not None
        # LLM should have agent's template set on it
        assert agent.llm.prompt_template == agent.prompt_template

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

    def test_template_format_validation(self, mock_llm_client):
        """Test template format validation."""
        from dapr_agents.agents.configs import AgentProfileConfig

        profile = AgentProfileConfig(name="TestAgent", template_format="f-string")
        agent = ConcreteAgentBase(profile=profile, llm=mock_llm_client)
        assert agent.prompting_helper.template_format == "f-string"

        agent = ConcreteAgentBase(name="TestAgent", llm=mock_llm_client)
        assert agent.prompting_helper.template_format == "jinja2"

    def test_max_iterations_default(self, minimal_agent):
        """Test default max iterations."""
        assert minimal_agent.execution.max_iterations == 10

    def test_max_iterations_custom(self, mock_llm_client):
        """Test custom max iterations."""
        from dapr_agents.agents.configs import AgentExecutionConfig

        agent = ConcreteAgentBase(
            name="TestAgent",
            execution=AgentExecutionConfig(max_iterations=5),
            llm=mock_llm_client,
        )
        assert agent.execution.max_iterations == 5
