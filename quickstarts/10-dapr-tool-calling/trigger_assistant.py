#!/usr/bin/env python3

"""
AssistantAgent Trigger Example

This script demonstrates how to trigger the AssistantAgent service
once it's running. The AssistantAgent runs as a Dapr workflow service
and can be triggered via various methods.

Usage:
    # Terminal 1: Start the AssistantAgent service
    python assistant_agent_example.py --provider openai

    # Terminal 2: Trigger the agent
    python trigger_assistant.py
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path


async def trigger_assistant_via_http(message: str, port: int = 8002):
    """
    Trigger the AssistantAgent via HTTP REST API.

    Note: This is a simplified example. The actual trigger mechanism
    depends on how the AssistantAgent service is configured.
    """

    url = f"http://localhost:{port}/start-workflow"

    payload = {"task": message, "session_id": "openai-test"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status in [200, 202]:  # 202 = Accepted (workflow started)
                    result = await response.json()
                    print("✅ Workflow started successfully!")
                    print(
                        f"   Instance ID: {result.get('workflow_instance_id', 'N/A')}"
                    )
                    print(f"   Message: {result.get('message', 'N/A')}")
                    return result
                else:
                    print(f"❌ Error: HTTP {response.status}")
                    print(f"Response: {await response.text()}")
                    return None

    except aiohttp.ClientConnectorError:
        print(
            f"❌ Connection failed. Is the AssistantAgent service running on port {port}?"
        )
        print("   Start it with: python assistant_agent_example.py --provider openai")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def check_chat_history(port: int = 8002, session_id: str = "demo-session"):
    """
    Check the chat history to see what responses were generated.
    """

    # Try to get chat history if there's an endpoint for it
    url = f"http://localhost:{port}/chat-history"

    payload = {"session_id": session_id}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print("📜 Chat History:")
                    for i, msg in enumerate(result.get("messages", []), 1):
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        print(f"   {i}. {role}: {content[:100]}...")
                    return result
                else:
                    print(
                        f"⚠️  Chat history endpoint not available (HTTP {response.status})"
                    )
                    return None

    except Exception as e:
        print(f"⚠️  Could not retrieve chat history: {e}")
        return None


async def trigger_assistant_via_dapr_workflow():
    """
    Trigger the AssistantAgent via Dapr workflow API.

    This is the proper way to trigger workflow-based agents.
    """

    # This would use the Dapr Python SDK to trigger workflows
    # For now, this is a placeholder showing the concept

    print("🔄 Triggering via Dapr Workflow API...")
    print("   (This requires additional Dapr workflow setup)")

    # Example of what this would look like:
    # from dapr.clients import DaprClient
    #
    # with DaprClient() as client:
    #     response = client.start_workflow(
    #         workflow_component="workflowstatestore",
    #         workflow_name="assistant_workflow",
    #         input={"message": message}
    #     )

    return {"status": "workflow_trigger_placeholder"}


async def main():
    """Main function to demonstrate different trigger methods."""

    print("🤖 AssistantAgent Trigger Demo")
    print("=" * 40)

    # Test message
    test_message = "What's the weather in San Francisco and calculate 15 * 8?"

    print(f"📝 Test message: {test_message}")
    print()

    # Method 1: HTTP trigger (simplified)
    print("🌐 Method 1: HTTP Trigger (simplified)")
    result1 = await trigger_assistant_via_http(test_message)
    print()

    # Wait a moment for workflow to complete
    if result1:
        print("⏳ Waiting for workflow to complete...")
        await asyncio.sleep(3)

        # Check chat history to see results
        print("🔍 Checking chat history for results...")
        await check_chat_history()
        print()

    # Method 2: Dapr workflow trigger (proper way)
    print("⚙️  Method 2: Dapr Workflow Trigger (proper way)")
    result2 = await trigger_assistant_via_dapr_workflow()
    print(result2)

    # Instructions
    print("📋 Instructions:")
    print("1. Start AssistantAgent service:")
    print("   python assistant_agent_example.py --provider echo  # For testing")
    print("   python assistant_agent_example.py --provider openai  # For production")
    print()
    print("2. Trigger the assistant:")
    print("   python trigger_assistant.py  # This script!")
    print()
    print("3. The workflow will execute in the background.")
    print("   Check the service logs to see tool execution results.")


if __name__ == "__main__":
    asyncio.run(main())
