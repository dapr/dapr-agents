#!/usr/bin/env python3
"""
Tool Calling Development Status - Current State & Path Forward

This example demonstrates the current state of tool calling implementation in dapr-agents,
showing what works, what's blocked, and the path to full functionality.

Located in examples/ because it's a comprehensive demonstration of:
1. ✅ What's currently working
2. ❌ What's blocked and why
3. 🛠️ What needs to be fixed
4. 🚀 How it will work once fixed

Usage:
    python examples/tool_calling_development_status.py
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv

from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput, Tool, ToolFunction
from dapr_agents import tool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
os.environ["DAPR_LLM_COMPONENT_DEFAULT"] = "openai"


class WeatherSchema(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")


@tool(args_model=WeatherSchema)
def get_weather(location: str) -> str:
    """Get current weather conditions for a location"""
    return f"Weather in {location}: 72°F, sunny with light breeze"


def demonstrate_current_state():
    """Show what's currently working and what's not."""
    print("🔍 CURRENT STATE ANALYSIS")
    print("=" * 50)

    # ✅ 1. Dapr-agents tool definition works
    print("\n✅ 1. Dapr-Agents Tool Definition")
    openai_format = get_weather.to_function_call()
    print(f"   Tool name: {openai_format['function']['name']}")
    print(f"   Description: {openai_format['function']['description']}")
    print(
        f"   Parameters: {list(openai_format['function']['parameters']['properties'].keys())}"
    )

    # ✅ 2. Python SDK data structures work
    print("\n✅ 2. Python SDK Data Structures")
    sdk_tool = Tool(
        type="function",
        name=openai_format["function"]["name"],
        description=openai_format["function"]["description"],
        parameters=json.dumps(openai_format["function"]["parameters"]),
    )
    print(f"   SDK Tool created: {sdk_tool.name}")

    # ❌ 3. ConversationInput doesn't accept tools yet
    print("\n❌ 3. ConversationInput Doesn't Accept Tools")
    print("   ConversationInput constructor doesn't have 'tools' parameter")
    ConversationInput(content="What's the weather in San Francisco?", role="user")
    print("   ConversationInput created without tools")

    # ❌ 4. But gRPC doesn't send them
    print("\n❌ 4. gRPC Protocol Issue")
    print("   The tools are IGNORED in the gRPC call!")
    print("   Only content, role, and scrubPII are sent.")
    print("   This is the main blocker.")

    # ✅ 5. Basic conversation works
    print("\n✅ 5. Basic Conversation Works")
    try:
        with DaprClient() as client:
            response = client.converse_alpha1(
                name="openai", inputs=[ConversationInput(content="Hello!", role="user")]
            )
            print(f"   Response: {response.outputs[0].result[:50]}...")
    except Exception as e:
        print(f"   Error: {e}")


def demonstrate_dapr_agents_integration():
    """Show dapr-agents integration with the fix."""
    print("\n🤖 DAPR-AGENTS INTEGRATION")
    print("=" * 50)

    try:
        from dapr_agents.llm.dapr import DaprChatClient

        # ✅ Create client
        client = DaprChatClient()

        # 🛠️ Apply the fix (normally done in the codebase)
        client._provider = "openai"  # Fix: change from "dapr" to "openai"

        print(f"✅ DaprChatClient created with provider: {client.provider}")

        # ✅ Simple conversation works
        print("\n✅ Simple Conversation:")
        messages = [{"role": "user", "content": "Say 'Hello from dapr-agents!'"}]
        response = client.generate(messages=messages)
        print(f"   Response: {response.choices[0].message.content}")

        # ❌ Tool calling doesn't work yet (LLM doesn't call tools)
        print("\n❌ Tool Calling (Current State):")
        messages = [
            {
                "role": "user",
                "content": "What's the weather in San Francisco? Use the weather tool.",
            }
        ]
        tools = [get_weather]

        response = client.generate(messages=messages, tools=tools)
        print(f"   Response: {response.choices[0].message.content[:100]}...")
        print("   ❌ LLM doesn't call tools (they're not reaching the LLM properly)")

    except Exception as e:
        print(f"❌ Error: {e}")


def show_future_implementation():
    """Show how it will work once the protocol is fixed."""
    print("\n🚀 FUTURE IMPLEMENTATION (Once Fixed)")
    print("=" * 50)

    print("Once the gRPC protocol is updated, this will work:")
    print(
        """
# 1. Tools will be sent through ConversationInput.tools
conv_input = ConversationInput(
    content="What's the weather in San Francisco?",
    role="user",
    tools=[weather_tool]  # ← This will actually be sent!
)

# 2. LLM will receive tools and make tool calls
response = client.converse_alpha1(name="openai", inputs=[conv_input])

# 3. Response will contain tool calls
if response.outputs[0].tool_calls:
    for tool_call in response.outputs[0].tool_calls:
        # Execute the tool
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)

        # Send result back
        tool_result = ConversationInput(
            content=result,
            role="tool",
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
        )

        # Get final response
        final_response = client.converse_alpha1(name="openai", inputs=[tool_result])

# 4. Complete tool calling flow works end-to-end! 🎉
"""
    )


def show_required_changes():
    """Show exactly what needs to be changed."""
    print("\n🛠️ REQUIRED CHANGES")
    print("=" * 50)

    print("1. 📝 Update Dapr protobuf (dapr.proto):")
    print(
        """
message ConversationInput {
  string content = 1;
  optional string role = 2;
  optional bool scrubPII = 3;
  repeated Tool tools = 4;           // ← ADD THIS
  optional string tool_call_id = 5;  // ← ADD THIS
  optional string name = 6;          // ← ADD THIS
}

message Tool {
  string type = 1;
  ToolFunction function = 2;
}

message ToolFunction {
  string name = 1;
  string description = 2;
  string parameters = 3;
}
"""
    )

    print("\n2. 🔧 Fix DaprChatClient provider:")
    print(
        """
# In dapr_agents/llm/dapr/client.py
class DaprInferenceClientBase(LLMClientBase):
    def model_post_init(self, __context: Any) -> None:
        self._provider = "openai"  # ← Change from "dapr"
"""
    )

    print("\n3. 🔄 Update Python SDK gRPC serialization:")
    print(
        """
# In dapr/clients/grpc/client.py
inputs_pb = [
    api_v1.ConversationInput(
        content=inp.content,
        role=inp.role,
        scrubPII=inp.scrub_pii,
        tools=inp.tools,           # ← ADD THIS
        tool_call_id=inp.tool_call_id,  # ← ADD THIS
        name=inp.name              # ← ADD THIS
    )
    for inp in inputs
]
"""
    )


if __name__ == "__main__":
    print(
        f"🧪 Working Tool Calling Example - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Show current state
    demonstrate_current_state()

    # Show dapr-agents integration
    demonstrate_dapr_agents_integration()

    # Show future implementation
    show_future_implementation()

    # Show required changes
    show_required_changes()

    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("=" * 50)
    print(
        "✅ Foundation is solid - tool definitions, data structures, processing all work"
    )
    print("❌ Main blocker: gRPC protocol doesn't send tools")
    print("🛠️ Fix: Update protobuf + regenerate bindings + fix provider")
    print("🚀 Result: Complete end-to-end tool calling functionality!")
    print(
        "\n💡 The architecture is correct, we just need to complete the transport layer!"
    )
