#!/usr/bin/env python3

"""
Test ChatCompletion model_dump format

This script tests what the ChatCompletion.model_dump() output looks like.
"""

import json
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents import tool


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location."""
    return f"The weather in {location} is sunny with a temperature of 72°F"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def test_model_dump():
    """Test ChatCompletion.model_dump() format."""

    print("🧪 Testing ChatCompletion.model_dump() Format")
    print("=" * 60)

    # Initialize DaprChatClient
    client = DaprChatClient(component_name="openai")

    # Define tools
    tools = [get_weather, calculate]

    # Test message that should trigger tool calls
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo and calculate 15 * 8?"}
    ]

    try:
        print("🚀 Calling DaprChatClient.generate()...")
        response = client.generate(messages=messages, tools=tools, tool_choice="auto")

        print("✅ Response received!")
        print(f"Response type: {type(response)}")
        print()

        # Convert to dictionary using model_dump()
        response_dict = response.model_dump()

        print("📋 model_dump() output:")
        print(json.dumps(response_dict, indent=2))
        print()

        # Test the AssistantAgent's access pattern
        print("🔍 Testing AssistantAgent access patterns:")

        # get_finish_reason pattern
        choices = response_dict.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason", None)
            print(f"   finish_reason: {finish_reason}")

        # get_tool_calls pattern
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls")
            print(f"   tool_calls type: {type(tool_calls)}")
            print(f"   tool_calls: {tool_calls}")

            if tool_calls:
                print(f"   Number of tool calls: {len(tool_calls)}")
                for i, tool_call in enumerate(tool_calls):
                    print(f"     {i+1}. Type: {type(tool_call)}")
                    print(f"        Content: {tool_call}")

                    # Test the execute_tool access pattern
                    if isinstance(tool_call, dict):
                        function_details = tool_call.get("function", {})
                        function_name = function_details.get("name")
                        print(f"        Function name: {function_name}")
                    else:
                        print(
                            f"        Tool call is not a dict! It's: {type(tool_call)}"
                        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model_dump()
