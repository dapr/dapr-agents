#!/usr/bin/env python3

"""
Test script to check if streaming functionality is available in the current Python SDK.
"""

import sys
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput


def test_streaming_availability():
    """Test if streaming methods are available."""
    print("🔍 Checking streaming availability...")

    # Check if the method exists
    client = DaprClient()

    if hasattr(client, "converse_stream_alpha1"):
        print("✅ converse_stream_alpha1 method found!")

        # Try to call it with echo component
        try:
            print("🧪 Testing streaming with echo component...")

            with DaprClient() as d:
                inputs = [
                    ConversationInput(content="Test streaming message", role="user")
                ]

                chunks_received = 0
                content_parts = []

                for chunk in d.converse_stream_alpha1(
                    name="echo", inputs=inputs, context_id="streaming-test-123"
                ):
                    chunks_received += 1

                    if (
                        hasattr(chunk, "result")
                        and chunk.result
                        and hasattr(chunk.result, "result")
                    ):
                        content = chunk.result.result
                        content_parts.append(content)
                        print(f"📦 Chunk {chunks_received}: '{content}'")

                    if hasattr(chunk, "context_id") and chunk.context_id:
                        print(f"🆔 Context ID: {chunk.context_id}")

                    if hasattr(chunk, "usage") and chunk.usage:
                        print(f"📊 Usage: {chunk.usage.total_tokens} tokens")

                print("\n✅ Streaming test successful!")
                print(f"   • Received {chunks_received} chunks")
                print(f"   • Full content: {''.join(content_parts)}")

                return True

        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
            return False
    else:
        print("❌ converse_stream_alpha1 method not found")
        print("Available methods:")
        for attr in dir(client):
            if "converse" in attr.lower():
                print(f"   • {attr}")
        return False


if __name__ == "__main__":
    success = test_streaming_availability()
    sys.exit(0 if success else 1)
