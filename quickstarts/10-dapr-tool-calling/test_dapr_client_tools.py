#!/usr/bin/env python3

"""
Test DaprChatClient Tool Calling

This script tests if the DaprChatClient can properly call tools with the OpenAI provider.
"""

import asyncio
import logging
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather information
    """
    return f"The weather in {location} is sunny with a temperature of 72°F"


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
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


async def test_dapr_client_tools():
    """Test DaprChatClient with tools."""

    print("🧪 Testing DaprChatClient Tool Calling")
    print("=" * 50)

    # Initialize DaprChatClient
    client = DaprChatClient(component_name="openai")

    # Define tools
    tools = [get_weather, calculate]

    # Test message that should trigger tool calls
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo and calculate 15 * 8?"}
    ]

    print(f"📝 Test message: {messages[0]['content']}")
    print(f"🔧 Available tools: {[tool.name for tool in tools]}")
    print()

    try:
        print("🚀 Calling DaprChatClient.generate()...")
        response = client.generate(
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Test with tool_choice parameter
        )

        print("✅ Response received!")
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")

        # Check if it's a ChatCompletion object
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message

            print("\n📋 Response Details:")
            print(f"   Finish reason: {choice.finish_reason}")
            print(f"   Content: {message.content}")

            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"   Tool calls: {len(message.tool_calls)}")
                for i, tool_call in enumerate(message.tool_calls):
                    print(
                        f"     {i+1}. {tool_call.function.name}({tool_call.function.arguments})"
                    )
            else:
                print("   ⚠️  No tool calls found")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_dapr_client_tools())
