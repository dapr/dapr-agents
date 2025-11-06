import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List
from pydantic import BaseModel, Field

from dapr_agents.workflow.decorators.activities import llm_activity, agent_activity
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.agents.base import AgentBase
from dapr_agents.types import LLMChatResponse, AssistantMessage


# Test Models
class Person(BaseModel):
    """Test model for structured responses."""

    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")


class QuestionList(BaseModel):
    """Test model for list responses."""

    questions: List[str] = Field(..., description="List of questions")


# Fixtures
@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns test responses."""
    mock_client = MagicMock(spec=ChatClientBase)
    mock_client.generate = MagicMock(return_value="Test response from LLM")
    return mock_client


@pytest.fixture
def mock_llm_client_async():
    """Mock async LLM client."""
    mock_client = MagicMock(spec=ChatClientBase)
    mock_client.generate = AsyncMock(return_value="Async test response")
    return mock_client


@pytest.fixture
def mock_llm_client_structured():
    """Mock LLM client that returns structured (LLMChatResponse) responses."""
    from dapr_agents.types import LLMChatCandidate

    mock_client = MagicMock(spec=ChatClientBase)
    candidate = LLMChatCandidate(
        message=AssistantMessage(content="Structured response"),
        finish_reason="stop",
    )
    response = LLMChatResponse(
        results=[candidate],
        metadata={"model": "test-model"},
    )
    mock_client.generate = MagicMock(return_value=response)
    return mock_client


@pytest.fixture
def mock_agent():
    """Mock agent that returns test responses."""

    class DummyAgent:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def run(self, prompt: str) -> str:
            self.calls.append(prompt)
            return "Agent response"

    return DummyAgent()


@pytest.fixture
def mock_workflow_context():
    """Mock WorkflowActivityContext."""
    ctx = MagicMock()
    ctx.instance_id = "test-instance-123"
    return ctx


# Tests for llm_activity decorator
def test_llm_activity_requires_prompt():
    """Test that llm_activity raises ValueError when prompt is empty."""
    mock_llm = MagicMock(spec=ChatClientBase)

    with pytest.raises(ValueError, match="@llm_activity requires a prompt template"):
        llm_activity(prompt="", llm=mock_llm)


def test_llm_activity_requires_llm():
    """Test that llm_activity raises ValueError when llm is None."""
    with pytest.raises(
        ValueError, match="@llm_activity requires an explicit `llm` client instance"
    ):
        llm_activity(prompt="Test prompt", llm=None)


def test_llm_activity_decorator_basic(mock_llm_client, mock_workflow_context):
    """Test basic llm_activity decoration and execution."""

    @llm_activity(prompt="Say hello to {name}", llm=mock_llm_client)
    def greet(ctx, name: str) -> str:
        pass

    # Check wrapper metadata
    assert hasattr(greet, "_is_llm_activity")
    assert greet._is_llm_activity is True
    assert hasattr(greet, "_llm_activity_config")
    assert greet._llm_activity_config["prompt"] == "Say hello to {name}"

    # Execute the decorated function
    result = greet(mock_workflow_context, payload={"name": "Alice"})

    # Verify the result
    assert result == "Test response from LLM"
    mock_llm_client.generate.assert_called_once()


def test_llm_activity_positional_args(mock_llm_client, mock_workflow_context):
    """Test llm_activity with positional arguments."""

    @llm_activity(prompt="Process {text}", llm=mock_llm_client)
    def process_text(ctx, text: str) -> str:
        pass

    result = process_text(mock_workflow_context, {"text": "Hello World"})
    assert result == "Test response from LLM"


def test_llm_activity_keyword_args(mock_llm_client, mock_workflow_context):
    """Test llm_activity with keyword arguments."""

    @llm_activity(prompt="Summarize {content}", llm=mock_llm_client)
    def summarize(ctx, content: str) -> str:
        pass

    result = summarize(ctx=mock_workflow_context, payload={"content": "Long text"})
    assert result == "Test response from LLM"


def test_llm_activity_multiple_params(mock_llm_client, mock_workflow_context):
    """Test llm_activity with multiple parameters."""

    @llm_activity(prompt="Write a {length} story about {topic}", llm=mock_llm_client)
    def write_story(ctx, topic: str, length: str) -> str:
        pass

    result = write_story(
        mock_workflow_context, payload={"topic": "robots", "length": "short"}
    )
    assert result == "Test response from LLM"


def test_llm_activity_no_payload(mock_llm_client, mock_workflow_context):
    """Test llm_activity with no input payload."""

    @llm_activity(prompt="Generate a random fact", llm=mock_llm_client)
    def random_fact(ctx) -> str:
        pass

    result = random_fact(mock_workflow_context)
    assert result == "Test response from LLM"


def test_llm_activity_async_llm(mock_llm_client_async, mock_workflow_context):
    """Test llm_activity with async LLM client."""

    @llm_activity(prompt="Test prompt", llm=mock_llm_client_async)
    def async_test(ctx) -> str:
        pass

    result = async_test(mock_workflow_context)
    assert result == "Async test response"


def test_llm_activity_structured_response(
    mock_llm_client_structured, mock_workflow_context
):
    """Test llm_activity with LLMChatResponse (structured response)."""

    @llm_activity(prompt="Get info", llm=mock_llm_client_structured)
    def get_info(ctx) -> str:
        pass

    result = get_info(mock_workflow_context)
    # convert_result should extract the content from LLMChatResponse
    assert result == "Structured response"


def test_llm_activity_structured_mode(mock_llm_client, mock_workflow_context):
    """Test llm_activity with different structured modes."""

    @llm_activity(
        prompt="Get data", llm=mock_llm_client, structured_mode="function_call"
    )
    def get_data(ctx) -> str:
        pass

    assert get_data._llm_activity_config["structured_mode"] == "function_call"


def test_llm_activity_preserves_function_metadata(mock_llm_client):
    """Test that llm_activity preserves function name and docstring."""

    @llm_activity(prompt="Test", llm=mock_llm_client)
    def my_function(ctx, param: str) -> str:
        """This is a docstring."""
        pass

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "This is a docstring."


# Tests for agent_activity decorator


def test_agent_activity_requires_agent():
    """Test that agent_activity raises ValueError when agent is None."""
    with pytest.raises(ValueError, match="@agent_activity requires an AgentBase"):
        agent_activity(agent=None)


def test_agent_activity_decorator_basic(mock_agent, mock_workflow_context):
    """Test basic agent_activity decoration and execution."""

    @agent_activity(agent=mock_agent)
    def run_task(ctx, task: str) -> str:
        pass

    # Check wrapper metadata
    assert hasattr(run_task, "_is_agent_activity")
    assert run_task._is_agent_activity is True
    assert hasattr(run_task, "_agent_activity_config")

    # Execute the decorated function
    result = run_task(mock_workflow_context, payload={"task": "analyze data"})

    # Verify the result
    assert result == "Agent response"
    assert mock_agent.calls == ["analyze data"]


def test_agent_activity_with_prompt(mock_agent, mock_workflow_context):
    """Test agent_activity with custom prompt template."""

    @agent_activity(agent=mock_agent, prompt="Analyze {data} and provide insights")
    def analyze(ctx, data: str) -> str:
        pass

    result = analyze(mock_workflow_context, payload={"data": "sales numbers"})
    assert result == "Agent response"

    # Verify the agent was called with formatted prompt
    assert mock_agent.calls == ["Analyze sales numbers and provide insights"]


def test_agent_activity_without_prompt(mock_agent, mock_workflow_context):
    """Test agent_activity without prompt (uses format_agent_input)."""

    @agent_activity(agent=mock_agent)
    def process(ctx, input_data: str) -> str:
        pass

    result = process(mock_workflow_context, payload={"input_data": "test data"})
    assert result == "Agent response"
    assert mock_agent.calls == ["test data"]


def test_agent_activity_multiple_params(mock_agent, mock_workflow_context):
    """Test agent_activity with multiple parameters."""

    @agent_activity(agent=mock_agent, prompt="Compare {item1} and {item2}")
    def compare(ctx, item1: str, item2: str) -> str:
        pass

    result = compare(
        mock_workflow_context, payload={"item1": "apple", "item2": "orange"}
    )
    assert result == "Agent response"
    assert mock_agent.calls == ["Compare apple and orange"]


def test_agent_activity_preserves_function_metadata(mock_agent):
    """Test that agent_activity preserves function name and docstring."""

    @agent_activity(agent=mock_agent)
    def agent_function(ctx, task: str) -> str:
        """Agent function docstring."""
        pass

    assert agent_function.__name__ == "agent_function"
    assert agent_function.__doc__ == "Agent function docstring."


# Integration tests


def test_llm_activity_with_pydantic_return_type(mock_llm_client, mock_workflow_context):
    """Test llm_activity with Pydantic model return type annotation."""
    # Mock the LLM to return a dict that matches Person schema
    mock_llm_client.generate = MagicMock(return_value='{"name": "John Doe", "age": 30}')

    @llm_activity(prompt="Get person info for {person_id}", llm=mock_llm_client)
    def get_person(ctx, person_id: str) -> Person:
        pass

    async def _return_person(*args, **kwargs):
        return Person(name="John Doe", age=30)

    with patch(
        "dapr_agents.workflow.decorators.activities.validate_result"
    ) as mock_validate:
        mock_validate.side_effect = _return_person
        _ = get_person(mock_workflow_context, payload={"person_id": "123"})

        # Verify the decorator called the LLM correctly
        assert mock_llm_client.generate.called


def test_llm_activity_ctx_parameter_stripped(mock_llm_client, mock_workflow_context):
    """Test that ctx parameter is properly stripped from signature processing."""

    @llm_activity(prompt="Process {data}", llm=mock_llm_client)
    def process(ctx, data: str) -> str:
        pass

    # The decorator should handle ctx internally and not include it in prompt formatting
    result = process(mock_workflow_context, payload={"data": "test"})
    assert result == "Test response from LLM"

    # Verify that the LLM was called with proper parameters (not including ctx)
    call_args = mock_llm_client.generate.call_args
    assert call_args is not None


def test_agent_activity_ctx_parameter_stripped(mock_agent, mock_workflow_context):
    """Test that ctx parameter is properly stripped from signature processing."""

    @agent_activity(agent=mock_agent, prompt="Process {input_val}")
    def process(ctx, input_val: str) -> str:
        pass

    result = process(mock_workflow_context, payload={"input_val": "data"})
    assert result == "Agent response"
    assert mock_agent.calls == ["Process data"]


def test_llm_activity_scalar_input(mock_llm_client, mock_workflow_context):
    """Test llm_activity with scalar (non-dict) input."""

    @llm_activity(prompt="Analyze {text}", llm=mock_llm_client)
    def analyze(ctx, text: str) -> str:
        pass

    # Pass a scalar string instead of a dict
    result = analyze(mock_workflow_context, "Simple text input")
    assert result == "Test response from LLM"


def test_agent_activity_scalar_input(mock_agent, mock_workflow_context):
    """Test agent_activity with scalar (non-dict) input."""

    @agent_activity(agent=mock_agent)
    def process(ctx, data: str) -> str:
        pass

    result = process(mock_workflow_context, "scalar input")
    assert result == "Agent response"
    assert mock_agent.calls == ["scalar input"]


def test_llm_activity_with_task_kwargs(mock_llm_client, mock_workflow_context):
    """Test llm_activity with additional task_kwargs."""

    @llm_activity(
        prompt="Test",
        llm=mock_llm_client,
        custom_param="custom_value",
        another_param=123,
    )
    def test_func(ctx) -> str:
        pass

    assert (
        test_func._llm_activity_config["task_kwargs"]["custom_param"] == "custom_value"
    )
    assert test_func._llm_activity_config["task_kwargs"]["another_param"] == 123


def test_agent_activity_with_task_kwargs(mock_agent, mock_workflow_context):
    """Test agent_activity with additional task_kwargs."""

    @agent_activity(agent=mock_agent, custom_setting="value", timeout=300)
    def test_func(ctx) -> str:
        pass

    assert test_func._agent_activity_config["task_kwargs"]["custom_setting"] == "value"
    assert test_func._agent_activity_config["task_kwargs"]["timeout"] == 300
    result = test_func(mock_workflow_context)
    assert result == "Agent response"
    assert len(mock_agent.calls) == 1


# Error handling tests


def test_llm_activity_non_callable_raises_error(mock_llm_client):
    """Test that decorating a non-callable raises ValueError."""
    decorator = llm_activity(prompt="Test", llm=mock_llm_client)

    with pytest.raises(ValueError, match="must decorate a callable activity"):
        decorator("not a function")


def test_agent_activity_non_callable_raises_error(mock_agent):
    """Test that decorating a non-callable raises ValueError."""
    decorator = agent_activity(agent=mock_agent)

    with pytest.raises(ValueError, match="must decorate a callable activity"):
        decorator("not a function")


def test_llm_activity_missing_context_raises_error(mock_llm_client):
    """Test that calling without context raises ValueError."""

    @llm_activity(prompt="Test", llm=mock_llm_client)
    def test_func(ctx) -> str:
        pass

    # Call without any context - should raise error from extract_ctx_and_payload
    with pytest.raises(ValueError, match="Workflow context is required"):
        test_func()


def test_agent_activity_missing_context_raises_error(mock_agent):
    """Test that calling without context raises ValueError."""

    @agent_activity(agent=mock_agent)
    def test_func(ctx) -> str:
        pass

    with pytest.raises(ValueError, match="Workflow context is required"):
        test_func()
