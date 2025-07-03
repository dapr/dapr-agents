"""
Comprehensive chat scenario tests covering all combinations of streaming and tool calling.

This test suite validates:
1. Non-streaming chat (basic conversation)
2. Streaming chat (basic conversation with streaming)
3. Tool calling non-streaming (tools without streaming)
4. Tool calling with streaming (tools with streaming)

Provider-specific behaviors:
- Echo: Works with all scenarios, OpenAI-compatible streaming format
- Anthropic: Non-streaming works, streaming has known issues with "invalid delta text field type"
- OpenAI: Should work with all scenarios (requires API key)
"""

import pytest
from typing import List, Dict, Any
from dapr.clients import DaprClient
from dapr.clients.grpc._response import ConversationResponse
from dapr_agents.tool import tool


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather information
    """
    return f"Weather in {location}: 72°F, sunny, light breeze"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_time() -> str:
    """Get the current time.

    Returns:
        Current time as a formatted string
    """
    from datetime import datetime

    current_time = datetime.now()
    return f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"


class TestChatScenarios:
    """Test all chat scenarios using Dapr Python SDK."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test method with Dapr client."""
        self.dapr_client = DaprClient()

    def teardown_method(self):
        """Cleanup after each test."""
        if hasattr(self, "dapr_client"):
            self.dapr_client.close()

    @pytest.fixture
    def weather_tools(self) -> List[Dict[str, Any]]:
        """Weather tool definition for testing."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "GetWeather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

    def validate_non_streaming_response(self, response: ConversationResponse) -> None:
        """Validate structure of non-streaming response using SDK response object."""
        assert response is not None
        assert hasattr(response, "outputs")
        assert len(response.outputs) > 0

        # Check first output has result
        output = response.outputs[0]
        assert hasattr(output, "result")
        assert isinstance(output.result, str)

        # For tool calling responses, result can be empty (especially for Anthropic)
        # If result is empty, there should be tool_calls
        if len(output.result) == 0:
            assert (
                hasattr(output, "tool_calls")
                and len(getattr(output, "tool_calls", [])) > 0
            ), "Empty result should have tool calls"

    def validate_streaming_response(self, chunks: List[Dict[str, Any]]) -> None:
        """Validate streaming response chunks."""
        assert chunks is not None
        assert len(chunks) > 0

        # Should have at least one chunk with content
        has_content = False
        for chunk in chunks:
            if chunk.get("choices") and len(chunk["choices"]) > 0:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    has_content = True
                    break

        assert has_content, "Streaming response should contain content chunks"

    def validate_tool_calls(
        self, response_or_chunks, is_streaming: bool = False, provider: str = None
    ) -> bool:
        """Check if tool calls were made in the response with provider-specific handling."""
        if is_streaming:
            # For streaming, look for tool_call chunks in OpenAI format
            for chunk in response_or_chunks:
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    if delta.get("tool_calls"):
                        return True
            return False
        else:
            # For non-streaming, use SDK response object
            if hasattr(response_or_chunks, "outputs"):
                for output in response_or_chunks.outputs:
                    if (
                        hasattr(output, "tool_calls")
                        and len(getattr(output, "tool_calls", [])) > 0
                    ):
                        return True
            return False

    def test_comprehensive_scenario_matrix(self, dapr_runtime, weather_tools):
        """Test all 4 core scenarios across all providers using Dapr Python SDK."""
        # Dapr runtime should be available via fixture
        assert dapr_runtime is not None

        providers = ["echo", "echo-tools", "anthropic", "gemini", "openai"]

        results = {}

        for provider in providers:
            print(f"\n🧪 Testing provider: {provider}")
            results[provider] = {}

            try:
                # Test 1: Non-streaming chat
                print("    💬 Testing non-streaming chat...")
                response = self.dapr_client.invoke_conversation(
                    name=provider,
                    inputs=[{"role": "user", "content": "Hello! What is 2+2?"}],
                )

                if response:
                    self.validate_non_streaming_response(response)
                    results[provider]["non_streaming_chat"] = "✅ Working"
                    print("      ✅ Non-streaming chat successful")
                else:
                    results[provider]["non_streaming_chat"] = "❌ Failed"
                    print("      ❌ Non-streaming chat failed")

            except Exception as e:
                results[provider]["non_streaming_chat"] = f"❌ Error: {str(e)}"
                print(f"      ❌ Non-streaming chat error: {e}")

            try:
                # Test 2: Streaming chat
                print("    🌊 Testing streaming chat...")
                chunks = []
                stream_response = self.dapr_client.invoke_conversation(
                    name=provider,
                    inputs=[{"role": "user", "content": "Count from 1 to 5"}],
                    stream=True,
                )

                # Collect streaming chunks
                for chunk in stream_response:
                    chunks.append(chunk)

                if chunks:
                    self.validate_streaming_response(chunks)
                    results[provider]["streaming_chat"] = "✅ Working"
                    print(f"      ✅ Streaming chat successful ({len(chunks)} chunks)")
                else:
                    results[provider]["streaming_chat"] = "❌ Failed"
                    print("      ❌ Streaming chat failed")

            except Exception as e:
                results[provider]["streaming_chat"] = f"❌ Error: {str(e)}"
                print(f"      ❌ Streaming chat error: {e}")

            try:
                # Test 3: Tool calling non-streaming
                print("    🔧 Testing tool calling (non-streaming)...")
                tool_response = self.dapr_client.invoke_conversation(
                    name=provider,
                    inputs=[
                        {
                            "role": "user",
                            "content": "What's the weather in Boston? Use the weather tool.",
                        }
                    ],
                    tools=weather_tools,
                )

                if tool_response and self.validate_tool_calls(
                    tool_response, is_streaming=False, provider=provider
                ):
                    results[provider]["tool_calling_non_streaming"] = "✅ Working"
                    print("      ✅ Tool calling successful")
                else:
                    results[provider]["tool_calling_non_streaming"] = "❌ Failed"
                    print("      ❌ Tool calling failed")

            except Exception as e:
                results[provider]["tool_calling_non_streaming"] = f"❌ Error: {str(e)}"
                print(f"      ❌ Tool calling error: {e}")

            try:
                # Test 4: Tool calling with streaming
                print("    🔧🌊 Testing tool calling with streaming...")
                tool_chunks = []
                tool_stream_response = self.dapr_client.invoke_conversation(
                    name=provider,
                    inputs=[
                        {
                            "role": "user",
                            "content": "What's the weather in San Francisco? Use the weather tool.",
                        }
                    ],
                    tools=weather_tools,
                    stream=True,
                )

                # Collect streaming chunks
                for chunk in tool_stream_response:
                    tool_chunks.append(chunk)

                if tool_chunks and self.validate_tool_calls(
                    tool_chunks, is_streaming=True, provider=provider
                ):
                    results[provider]["tool_calling_streaming"] = "✅ Working"
                    print(
                        f"      ✅ Tool calling + streaming successful ({len(tool_chunks)} chunks)"
                    )
                else:
                    results[provider]["tool_calling_streaming"] = "❌ Failed"
                    print("      ❌ Tool calling + streaming failed")

            except Exception as e:
                results[provider]["tool_calling_streaming"] = f"❌ Error: {str(e)}"
                print(f"      ❌ Tool calling + streaming error: {e}")

        # Print comprehensive results matrix
        print("\n📊 COMPREHENSIVE TEST RESULTS MATRIX")
        print("=" * 80)

        scenarios = [
            "non_streaming_chat",
            "streaming_chat",
            "tool_calling_non_streaming",
            "tool_calling_streaming",
        ]

        # Header
        print(
            f"{'Provider':<15} {'Non-Stream':<15} {'Streaming':<15} {'Tool-Call':<15} {'Tool+Stream':<15}"
        )
        print(f"{'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")

        # Results for each provider
        for provider in providers:
            row = f"{provider:<15}"
            for scenario in scenarios:
                status = results[provider].get(scenario, "❌ Skipped")
                # Truncate status for display
                display_status = status[:13] + ".." if len(status) > 15 else status
                row += f" {display_status:<15}"
            print(row)

        print("=" * 80)

        # Count successes
        total_tests = len(providers) * len(scenarios)
        successful_tests = 0

        for provider in providers:
            for scenario in scenarios:
                if results[provider].get(scenario, "").startswith("✅"):
                    successful_tests += 1

        success_rate = (successful_tests / total_tests) * 100
        print(
            f"🎯 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)"
        )

        # Validate we have some successful tests
        assert successful_tests > 0, "At least some tests should pass"

        # Ensure echo providers work (they don't require API keys)
        echo_success = 0
        for provider in ["echo", "echo-tools"]:
            for scenario in scenarios:
                if results[provider].get(scenario, "").startswith("✅"):
                    echo_success += 1

        assert (
            echo_success >= 4
        ), "Echo providers should have at least 4 successful tests"
