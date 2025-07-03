#!/usr/bin/env python3

"""
Enhanced DaprChatClient Tool Calling Example

This example demonstrates tool calling using the ENHANCED DaprChatClient that
now properly supports the parts format for multi-turn tool calling conversations.

This shows how the DaprChatClient can now handle:
1. Assistant messages with tool_calls using parts format
2. Tool result messages using ToolResultContent
3. Multi-turn conversations with proper message flow
"""

import argparse
import json
import os
from pathlib import Path
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.tool import tool
from dapr_agents.types.message import ToolMessage
from dapr.clients.grpc._request import ConversationInput, ContentPart, ToolResultContent
from dotenv import load_dotenv


# Load environment variables
def load_env_file():
    """Load environment variables from .env file at the root of the repo."""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded env from: {env_file}")


# Define tools
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
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def execute_tool_call(tool_call, available_tools):
    """Execute a single tool call and return the result."""
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    print(f"   🔧 Executing: {tool_name}({tool_args})")

    # Find the tool by name - check both the tool.name and function name
    for t in available_tools:
        # Check AgentTool.name (capitalized) or function name
        tool_matches = False
        if hasattr(t, "name") and t.name == tool_name:
            tool_matches = True
        elif hasattr(t, "__name__") and t.__name__ == tool_name:
            tool_matches = True
        elif hasattr(t, "__name__") and t.__name__.lower() == tool_name.lower():
            tool_matches = True

        if tool_matches:
            try:
                result = t(**tool_args)
                print(f"   ✅ Result: {result}")
                return ToolMessage(
                    tool_call_id=tool_call.get("id", f"call_{tool_name}"),
                    name=tool_name,
                    content=result,
                )
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"   ❌ {error_msg}")
                return ToolMessage(
                    tool_call_id=tool_call.get("id", f"call_{tool_name}"),
                    name=tool_name,
                    content=error_msg,
                )

    # Tool not found
    available_names = []
    for t in available_tools:
        if hasattr(t, "name"):
            available_names.append(f"tool.name='{t.name}'")
        if hasattr(t, "__name__"):
            available_names.append(f"__name__='{t.__name__}'")

    error_msg = f"Tool {tool_name} not found. Available: {available_names}"
    print(f"   ❌ {error_msg}")
    return ToolMessage(
        tool_call_id=tool_call.get("id", f"call_{tool_name}"),
        name=tool_name,
        content=error_msg,
    )


def run_enhanced_tool_calling_example(provider: str = "openai"):
    """Run enhanced tool calling example with the updated DaprChatClient."""

    print("🚀 Enhanced DaprChatClient Tool Calling Example")
    print("=" * 60)
    print(f"Provider: {provider}")
    print("This example uses the ENHANCED DaprChatClient with:")
    print("• ✅ Assistant messages with tool_calls using parts format")
    print("• ✅ Tool result messages using ToolResultContent")
    print("• ✅ Multi-turn conversations with proper message flow")
    print("• ✅ Raw response access for conversation building")
    print()

    # Check API key
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return

    # Initialize client and tools
    client = DaprChatClient()
    available_tools = [get_weather, calculate]

    # User's question
    user_message = "What's the weather like in Tokyo? Also calculate 15 * 8 + 7"
    print(f"👤 User: {user_message}")
    print()

    # Step 1: Send initial request with tools using generate_raw
    print("📤 Step 1: Sending query with tools to LLM (using generate_raw)...")
    messages = [{"role": "user", "content": user_message}]

    try:
        # Use generate_raw to get access to the actual response parts
        raw_response = client.generate_raw(
            messages=messages,
            tools=available_tools,
            llm_component=provider,
            stream=False,
        )

        # Extract information from raw response
        output = raw_response.outputs[0]
        print(f"📋 Response text: {output.result}")
        print(f"🏁 Finish reason: {output.finish_reason}")

        # Extract tool calls from response
        tool_calls = output.get_tool_calls()
        print(f"🔧 Tool calls: {len(tool_calls)} generated")
        print()

        if not tool_calls:
            print("ℹ️  No tool calls requested, LLM provided direct response")
            return

        # Step 2: Build conversation history using the enhanced DaprChatClient
        print(
            "📋 Step 2: Building conversation history with enhanced DaprChatClient..."
        )

        # Create assistant message with tool_calls (this will use the new parts format)
        assistant_message = {
            "role": "assistant",
            "content": output.result,  # May be None
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name if hasattr(tc, "name") else tc.function.name,
                        "arguments": tc.arguments
                        if hasattr(tc, "arguments")
                        else tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }

        print("   ✅ Assistant message created with tool_calls")

        # Step 3: Execute tools and create tool result messages
        print("🔧 Step 3: Execute tools and create tool result messages...")
        tool_results = []
        for i, tool_call in enumerate(tool_calls, 1):
            print(f"   Tool {i}/{len(tool_calls)}:")

            # Convert tool call to dict format for execute_tool_call
            tool_call_dict = {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.name
                    if hasattr(tool_call, "name")
                    else tool_call.function.name,
                    "arguments": tool_call.arguments
                    if hasattr(tool_call, "arguments")
                    else tool_call.function.arguments,
                },
            }

            result = execute_tool_call(tool_call_dict, available_tools)
            tool_results.append(result)

        print(f"   ✅ Executed {len(tool_calls)} tools")
        print()

        # Step 4: Build complete conversation and get final response
        print("💬 Step 4: Building complete conversation...")

        # Convert ToolMessage objects to dict format
        tool_result_dicts = [tool_result.model_dump() for tool_result in tool_results]

        # Build conversation history
        conversation = messages + [assistant_message] + tool_result_dicts

        print(f"📝 Conversation history: {len(conversation)} messages")
        for i, msg in enumerate(conversation):
            content = msg.get("content") or "None"
            if isinstance(content, str) and len(content) > 50:
                content = content[:50] + "..."
            print(f"   Message {i+1}: role={msg.get('role')}, content={content}")
            if msg.get("tool_calls"):
                print(f"      Tool calls: {len(msg['tool_calls'])}")
            if msg.get("tool_call_id"):
                print(f"      Tool call ID: {msg['tool_call_id']}")
        print()

        # Get final response using the enhanced DaprChatClient
        print("🤖 Getting final response with enhanced DaprChatClient...")
        final_response = client.generate(
            messages=conversation, llm_component=provider, stream=False
        )

        # Step 5: Show final response
        print("🎉 Final Response:")
        if (
            final_response
            and hasattr(final_response, "choices")
            and final_response.choices
        ):
            final_text = final_response.choices[0].message.content
            if final_text:
                print(f"🤖 Assistant: {final_text}")
            else:
                print(f"📋 Raw response: {final_response}")
        else:
            print("❌ No final response received")

        print()
        print("✅ Enhanced DaprChatClient tool calling workflow successful!")
        print("   ✅ Raw response access for proper conversation building")
        print("   ✅ Assistant messages with tool_calls using parts format")
        print("   ✅ Tool result messages using ToolResultContent")
        print("   ✅ Multi-turn conversation completed")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced DaprChatClient Tool Calling Example"
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Dapr conversation provider (openai, anthropic, etc.)",
    )

    args = parser.parse_args()

    print(
        """
🔧 Enhanced DaprChatClient Tool Calling Example

This example demonstrates the COMPLETE tool calling workflow using
the ENHANCED DaprChatClient that now supports:

• Assistant messages with tool_calls using parts format
• Tool result messages using ToolResultContent
• Raw response access for proper conversation building
• Multi-turn conversations with proper message flow

Available tools:
• get_weather(location) - Get weather for a location
• calculate(expression) - Perform mathematical calculations

Make sure Dapr is running with your components configured!
"""
    )

    # Load environment variables
    load_env_file()

    # Run the example
    run_enhanced_tool_calling_example(args.provider)


if __name__ == "__main__":
    main()
