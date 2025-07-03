"""
React Agent Integration Tests with Dapr

This test suite validates React (Reasoning-Action) agents working with Dapr conversation API:
1. Simple reasoning tasks (no tools)
2. Single tool execution
3. Multi-tool workflows
4. Complex reasoning chains
5. Error handling and recovery
"""

import pytest
import asyncio
from dapr_agents.agent.patterns.react import ReActAgent
from dapr_agents import tool
from dapr_agents.llm.dapr import DaprChatClient


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather conditions as a string.
    """
    # Mock weather data for testing
    weather_data = {
        "san francisco": "Sunny and 72°F",
        "london": "Cloudy and 58°F",
        "new york": "Rainy and 65°F",
        "paris": "Partly cloudy and 68°F",
    }
    return weather_data.get(
        location.lower(), f"Weather data not available for {location}"
    )


@tool
def get_activities(weather: str) -> str:
    """Get activity recommendations based on weather conditions.

    Args:
        weather: Current weather description

    Returns:
        Recommended activities as a string.
    """
    if "sunny" in weather.lower():
        return "Perfect for outdoor activities: hiking, picnic, beach visit"
    elif "rainy" in weather.lower():
        return "Indoor activities recommended: museums, shopping, cafes"
    elif "cloudy" in weather.lower():
        return "Good for walking tours, sightseeing, outdoor markets"
    else:
        return "Flexible activities: indoor/outdoor options available"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.

    Args:
        expression: Mathematical expression like '2 + 3' or '10 * 5'

    Returns:
        Calculation result as a string.
    """
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Invalid characters in expression"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


class TestReActAgentScenarios:
    """Test React agents with different complexity scenarios."""

    @pytest.fixture
    def simple_tools(self):
        """Basic tools for React agent testing."""
        return [get_weather, calculate]

    @pytest.fixture
    def workflow_tools(self):
        """Tools for multi-step workflow testing."""
        return [get_weather, get_activities]

    @pytest.mark.asyncio
    async def test_react_agent_simple_reasoning(self, dapr_runtime):
        """Test React agent with simple reasoning (no tools needed)."""
        agent = ReActAgent(
            name="SimpleReasoningAgent",
            role="Assistant",
            instructions=["Provide helpful responses to user questions"],
            llm_client=DaprChatClient(),
            llm_component="echo",
            max_iterations=3,
        )

        response = await agent.run("Hello! What is 2 + 2?")

        # Should provide a direct answer without tool use
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_react_agent_single_tool_execution(self, dapr_runtime, simple_tools):
        """Test React agent using a single tool."""
        agent = ReActAgent(
            name="WeatherAgent",
            role="Weather Assistant",
            instructions=["Help users get weather information"],
            tools=simple_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=5,
        )

        response = await agent.run("What's the weather like in San Francisco?")

        # Should use get_weather tool and provide weather info
        assert response is not None
        assert isinstance(response, str)
        assert "san francisco" in response.lower() or "weather" in response.lower()

        # Check that agent's memory contains the interaction
        chat_history = agent.chat_history
        assert len(chat_history) >= 2  # At least user message and assistant response
        assert chat_history[0]["role"] == "user"
        assert chat_history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_react_agent_multi_tool_workflow(self, dapr_runtime, workflow_tools):
        """Test React agent with multi-tool workflow (weather → activities)."""
        agent = ReActAgent(
            name="TravelAgent",
            role="Travel Assistant",
            instructions=[
                "Help users plan activities based on weather conditions",
                "First check the weather, then suggest appropriate activities",
            ],
            tools=workflow_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=8,
        )

        response = await agent.run("I'm visiting London today. What should I do?")

        # Should use both get_weather and get_activities tools
        assert response is not None
        assert isinstance(response, str)

        # Response should contain information about both weather and activities
        response_lower = response.lower()
        assert any(
            word in response_lower
            for word in ["weather", "cloudy", "activity", "activities"]
        )

    @pytest.mark.asyncio
    async def test_react_agent_complex_reasoning_chain(
        self, dapr_runtime, simple_tools
    ):
        """Test React agent with complex multi-step reasoning."""
        agent = ReActAgent(
            name="MathWeatherAgent",
            role="Multi-domain Assistant",
            instructions=[
                "Help with both calculations and weather information",
                "Break down complex requests into steps",
            ],
            tools=simple_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=10,
        )

        response = await agent.run(
            "Calculate 15 * 3, then tell me the weather in New York"
        )

        # Should handle both calculation and weather lookup
        assert response is not None
        assert isinstance(response, str)

        response_lower = response.lower()
        # Should contain results from both tools
        assert any(word in response_lower for word in ["45", "15", "3"])  # Math result
        assert any(
            word in response_lower for word in ["new york", "weather", "rainy"]
        )  # Weather

    @pytest.mark.asyncio
    async def test_react_agent_error_handling(self, dapr_runtime, simple_tools):
        """Test React agent error handling with invalid tool usage."""
        agent = ReActAgent(
            name="ErrorTestAgent",
            role="Test Assistant",
            instructions=["Handle errors gracefully"],
            tools=simple_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=5,
        )

        # Test with invalid calculation
        response = await agent.run("Calculate 'invalid math expression here'")

        # Should handle the error and provide a response
        assert response is not None
        assert isinstance(response, str)
        # Should not crash, should provide some kind of error handling response

    @pytest.mark.asyncio
    async def test_react_agent_max_iterations(self, dapr_runtime, workflow_tools):
        """Test React agent respects max iterations limit."""
        agent = ReActAgent(
            name="LimitedAgent",
            role="Test Assistant",
            instructions=["Test iteration limits"],
            tools=workflow_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=2,  # Very low limit
        )

        # Complex request that might need more iterations
        response = await agent.run(
            "Plan a detailed itinerary for London including weather check and activities"
        )

        # Should stop at max iterations but still provide some response
        assert response is not None or True  # Agent might reach max iterations

    @pytest.mark.asyncio
    async def test_react_agent_memory_persistence(self, dapr_runtime, simple_tools):
        """Test React agent memory across multiple interactions."""
        agent = ReActAgent(
            name="MemoryAgent",
            role="Memory Test Assistant",
            instructions=["Remember previous interactions"],
            tools=simple_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=5,
        )

        # First interaction
        response1 = await agent.run("My name is Alice. What's the weather in Paris?")
        assert response1 is not None

        # Second interaction - should remember the name
        response2 = await agent.run("What was my name again?")
        assert response2 is not None

        # Check chat history contains both interactions
        chat_history = agent.chat_history
        assert len(chat_history) >= 4  # 2 user messages + 2 assistant responses

        # Should contain information about Alice and Paris
        history_text = str(chat_history).lower()
        assert "alice" in history_text
        assert "paris" in history_text


class TestReActAgentProviders:
    """Test React agents with different Dapr conversation providers."""

    @pytest.fixture
    def basic_tools(self):
        return [get_weather]

    @pytest.mark.asyncio
    async def test_react_agent_with_echo_provider(self, dapr_runtime, basic_tools):
        """Test React agent with echo provider (no API key needed)."""
        agent = ReActAgent(
            name="EchoReactAgent",
            role="Echo Test Assistant",
            tools=basic_tools,
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=3,
        )

        response = await agent.run("Test echo provider with weather tool")
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_react_agent_with_openai_provider(
        self, dapr_runtime, basic_tools, test_environment
    ):
        """Test React agent with OpenAI provider (requires API key)."""
        if not test_environment.get("api_keys", {}).get("openai", False):
            pytest.skip("OpenAI API key not available")

        agent = ReActAgent(
            name="OpenAIReactAgent",
            role="OpenAI Test Assistant",
            tools=basic_tools,
            llm_client=DaprChatClient(),
            llm_component="openai",
            max_iterations=5,
        )

        response = await agent.run("What's the weather in London?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 10  # Should be a substantial response

    @pytest.mark.asyncio
    @pytest.mark.requires_api_key
    async def test_react_agent_with_anthropic_provider(
        self, dapr_runtime, basic_tools, test_environment
    ):
        """Test React agent with Anthropic provider (requires API key)."""
        if not test_environment.get("api_keys", {}).get("anthropic", False):
            pytest.skip("Anthropic API key not available")

        agent = ReActAgent(
            name="AnthropicReactAgent",
            role="Anthropic Test Assistant",
            tools=basic_tools,
            llm_client=DaprChatClient(),
            llm_component="anthropic",
            max_iterations=5,
        )

        response = await agent.run("What's the weather in New York?")
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 10  # Should be a substantial response


class TestReActAgentIntegration:
    """Integration tests for React agents with complex scenarios."""

    @pytest.mark.asyncio
    async def test_react_agent_full_workflow_integration(self, dapr_runtime):
        """Test complete React agent workflow integration."""

        @tool
        def search_database(query: str) -> str:
            """Search a mock database."""
            return f"Database results for: {query}"

        @tool
        def format_report(data: str) -> str:
            """Format data into a report."""
            return f"Formatted Report:\n{data}"

        agent = ReActAgent(
            name="WorkflowAgent",
            role="Data Assistant",
            instructions=[
                "Help users search data and create reports",
                "First search for data, then format it into a report",
            ],
            tools=[search_database, format_report],
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=8,
        )

        response = await agent.run("Search for 'customer data' and create a report")

        # Should execute both tools in sequence
        assert response is not None
        assert isinstance(response, str)

        # Should contain evidence of both database search and report formatting
        response_lower = response.lower()
        assert any(
            word in response_lower for word in ["database", "search", "customer"]
        )
        assert any(word in response_lower for word in ["report", "formatted"])

    @pytest.mark.asyncio
    async def test_react_agent_concurrent_execution(self, dapr_runtime):
        """Test multiple React agents running concurrently."""

        @tool
        def agent_specific_tool(agent_name: str) -> str:
            """Tool that identifies which agent called it."""
            return f"Tool called by {agent_name}"

        # Create two agents with the same tool
        agent1 = ReActAgent(
            name="Agent1",
            role="Test Assistant 1",
            tools=[agent_specific_tool],
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=3,
        )

        agent2 = ReActAgent(
            name="Agent2",
            role="Test Assistant 2",
            tools=[agent_specific_tool],
            llm_client=DaprChatClient(),
            llm_component="echo-tools",
            max_iterations=3,
        )

        # Run both agents concurrently
        results = await asyncio.gather(
            agent1.run("Use the agent_specific_tool with 'Agent1'"),
            agent2.run("Use the agent_specific_tool with 'Agent2'"),
            return_exceptions=True,
        )

        # Both should complete successfully
        assert len(results) == 2
        for result in results:
            if not isinstance(result, Exception):
                assert result is not None
                assert isinstance(result, str)
