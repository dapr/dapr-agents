#!/usr/bin/env python3

"""
Example demonstrating Dapr streaming chat completion using dapr-agents.

This example shows how to use the streaming capability with Dapr LLM components.

Prerequisites:
- Dapr sidecar running with LLM components
- Set DAPR_LLM_COMPONENT_DEFAULT environment variable
"""

import os
import json
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput, Tool, ToolFunction


def streaming_example():
    """Example of streaming chat completion with Dapr."""
    print("🚀 Dapr Streaming Chat Completion Example")

    # Initialize the Dapr client
    client = DaprClient()

    # Create conversation input using the proper dataclass
    inputs = [
        ConversationInput.from_text(
            text="Tell me a story about a robot learning to paint", role="user"
        )
    ]

    print("\n📡 Starting streaming conversation...")
    print("🤖 Assistant: ", end="", flush=True)

    try:
        # Generate streaming response
        full_content = ""
        total_tokens = 0

        for chunk in client.converse_stream_alpha1(
            name=os.getenv("DAPR_LLM_COMPONENT_DEFAULT", "echo"),
            inputs=inputs,
            temperature=0.7,
            context_id="streaming-example-123",
        ):
            if chunk.chunk and chunk.chunk.content:
                print(chunk.chunk.content, end="", flush=True)
                full_content += chunk.chunk.content

            if chunk.complete and chunk.complete.usage:
                total_tokens = chunk.complete.usage.total_tokens

        print(f"\n\n✅ Complete response received: {len(full_content)} characters")
        if total_tokens > 0:
            print(f"💰 Usage: {total_tokens} tokens")

        # Also show the total content received
        if full_content:
            print(f"\n📝 Total content: {len(full_content)} characters")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. Dapr sidecar is running")
        print("   2. DAPR_LLM_COMPONENT_DEFAULT environment variable is set")
        print("   3. LLM component is properly configured")
    finally:
        client.close()


def non_streaming_example():
    """Example of non-streaming chat completion with Dapr for comparison."""
    print("\n🔄 Non-streaming Chat Completion Example")

    # Initialize the Dapr client
    client = DaprClient()

    # Create conversation input using the proper dataclass
    inputs = [
        ConversationInput.from_text(text="What is the capital of France?", role="user")
    ]

    print("\n📞 Making non-streaming request...")

    try:
        # Generate non-streaming response
        response = client.converse_alpha1(
            name=os.getenv("DAPR_LLM_COMPONENT_DEFAULT", "echo"),
            inputs=inputs,
            temperature=0.3,
            context_id="non-streaming-example-456",
        )

        print(f"🤖 Assistant: {response.outputs[0].result}")

        if response.usage:
            print(f"💰 Usage: {response.usage.total_tokens} tokens")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()


def tool_calling_example():
    """Example of tool calling with streaming."""
    print("\n🔧 Tool Calling with Streaming Example")

    # Create proper Tool using the correct dataclass structure
    weather_tool = Tool(
        type="function",
        name="get_weather",
        description="Get current weather for a location",
        parameters=json.dumps(
            {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            }
        ),
    )

    client = DaprClient()

    inputs = [
        ConversationInput.from_text(
            text="What's the weather like in San Francisco?", role="user"
        )
    ]

    print("\n📞 Making tool calling request...")

    try:
        response = client.converse_alpha1(
            name=os.getenv("DAPR_LLM_COMPONENT_DEFAULT", "echo-tools"),
            inputs=inputs,
            tools=[weather_tool],
            context_id="tool-calling-example-789",
        )

        print(f"🤖 Assistant: {response.outputs[0].result}")

        if response.usage:
            print(f"💰 Usage: {response.usage.total_tokens} tokens")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have a tool-calling capable component configured")
    finally:
        client.close()


def main():
    """Main function to run examples."""
    print("=" * 60)
    print("  Dapr Streaming Chat Completion Examples")
    print("=" * 60)

    # Check environment
    component_name = os.getenv("DAPR_LLM_COMPONENT_DEFAULT")
    if not component_name:
        print("⚠️  Warning: DAPR_LLM_COMPONENT_DEFAULT not set")
        print("   Setting default to 'echo' for demo purposes")
        os.environ["DAPR_LLM_COMPONENT_DEFAULT"] = "echo"
        component_name = "echo"

    print(f"🎯 Using component: {component_name}")

    # Run streaming example
    streaming_example()

    # Run non-streaming example for comparison
    non_streaming_example()

    # Run tool calling example
    tool_calling_example()

    print("\n" + "=" * 60)
    print("  Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
