#!/usr/bin/env python3

"""
Comprehensive Dapr LLM Testing Suite

This comprehensive example demonstrates production-ready testing of Dapr LLM components
across multiple providers with automatic component creation and detailed diagnostics.

Unlike the quickstart examples (which focus on learning), this is a diagnostic tool for:
- Multi-provider testing (Echo, OpenAI, Anthropic)
- Component auto-creation and validation
- Production readiness checking
- API key validation and troubleshooting

Prerequisites:
1. Start Dapr sidecar: python tools/run_dapr_dev.py --build
2. Set environment variables for API keys (optional)
3. Components will be auto-created in ./components/ directory

Usage:
    python examples/comprehensive_dapr_llm_testing.py [--component echo|openai|anthropic]
    python examples/comprehensive_dapr_llm_testing.py --show-config
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput

# Load environment variables
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"📄 Loaded environment from: {env_file}")


def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")

    # Check if Dapr is running
    try:
        import requests

        response = requests.get("http://localhost:3500/v1.0/healthz", timeout=2)
        if response.status_code == 204:
            print("✅ Dapr sidecar is running")
        else:
            print("❌ Dapr sidecar not responding correctly")
            return False
    except Exception:
        print("❌ Dapr sidecar not running")
        print("💡 Start with: python tools/run_dapr_dev.py --build")
        return False

    return True


def test_echo_component():
    """Test the echo conversation component."""
    print("\n🔊 Testing Echo Component")
    print("=" * 40)
    print("Echo component simply returns the input - useful for testing")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput.from_text(
                    text="Hello from the echo component test!", role="user"
                )
            ]

            print("📤 Sending: Hello from the echo component test!")

            # Non-streaming test
            response = client.converse_alpha1(
                name="echo", inputs=inputs, context_id="echo-test-123"
            )

            print(f"📥 Received: {response.outputs[0].result}")
            print(f"🆔 Context ID: {response.context_id}")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs = [
                ConversationInput.from_text(
                    text="This is a streaming test with echo!", role="user"
                )
            ]
            print("📤 Streaming: This is a streaming test with echo!")
            print("📥 Streamed response: ", end="", flush=True)

            for chunk in client.converse_stream_alpha1(
                name="echo", inputs=inputs, context_id="echo-stream-456"
            ):
                if chunk.chunk and chunk.chunk.content:
                    print(chunk.chunk.content, end="", flush=True)

            print("\n✅ Echo component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ Echo component test failed: {e}")
        return False


def test_openai_component():
    """Test the OpenAI conversation component."""
    print("\n🤖 Testing OpenAI Component")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        print("💡 Set with: export OPENAI_API_KEY=your_key_here")
        return False

    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"✅ OpenAI API Key: {masked_key}")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput.from_text(
                    text="Write a haiku about programming", role="user"
                )
            ]

            print("📤 Sending: Write a haiku about programming")

            # Non-streaming test
            print("\n📡 Testing non-streaming...")
            response = client.converse_alpha1(
                name="openai",
                inputs=inputs,
                temperature=0.7,
                context_id="openai-test-123",
            )

            print(f"📥 Response: {response.outputs[0].result}")
            if response.usage:
                print(f"📊 Usage: {response.usage.total_tokens} tokens")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs = [
                ConversationInput.from_text(
                    text="Tell me a short joke about AI", role="user"
                )
            ]
            print("📤 Streaming: Tell me a short joke about AI")
            print("📥 Streamed response: ", end="", flush=True)

            total_tokens = 0
            for chunk in client.converse_stream_alpha1(
                name="openai",
                inputs=inputs,
                temperature=0.8,
                context_id="openai-stream-456",
            ):
                if chunk.chunk and chunk.chunk.content:
                    print(chunk.chunk.content, end="", flush=True)
                if chunk.complete and chunk.complete.usage:
                    total_tokens = chunk.complete.usage.total_tokens

            if total_tokens:
                print(f"\n📊 Total tokens used: {total_tokens}")

            print("\n✅ OpenAI component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ OpenAI component test failed: {e}")
        print("💡 Check your OpenAI API key and component configuration")
        return False


def test_anthropic_component():
    """Test the Anthropic conversation component."""
    print("\n🧠 Testing Anthropic Component")
    print("=" * 40)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        print("💡 Set with: export ANTHROPIC_API_KEY=your_key_here")
        print("💡 Create component: components/anthropic-conversation.yaml")
        return False

    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"✅ Anthropic API Key: {masked_key}")

    try:
        with DaprClient() as client:
            inputs = [
                ConversationInput.from_text(
                    text="Explain quantum computing in simple terms", role="user"
                )
            ]

            print("📤 Sending: Explain quantum computing in simple terms")

            # Non-streaming test
            print("\n📡 Testing non-streaming...")
            response = client.converse_alpha1(
                name="anthropic",
                inputs=inputs,
                temperature=0.7,
                context_id="anthropic-test-123",
            )

            print(f"📥 Response: {response.outputs[0].result}")
            if response.usage:
                print(f"📊 Usage: {response.usage.total_tokens} tokens")

            # Streaming test
            print("\n📡 Testing streaming...")
            inputs = [
                ConversationInput.from_text(
                    text="What are three benefits of renewable energy?", role="user"
                )
            ]
            print("📤 Streaming: What are three benefits of renewable energy?")
            print("📥 Streamed response: ", end="", flush=True)

            total_tokens = 0
            for chunk in client.converse_stream_alpha1(
                name="anthropic",
                inputs=inputs,
                temperature=0.8,
                context_id="anthropic-stream-456",
            ):
                if chunk.chunk and chunk.chunk.content:
                    print(chunk.chunk.content, end="", flush=True)
                if chunk.complete and chunk.complete.usage:
                    total_tokens = chunk.complete.usage.total_tokens

            if total_tokens:
                print(f"\n📊 Total tokens used: {total_tokens}")

            print("\n✅ Anthropic component test completed successfully")
            return True

    except Exception as e:
        print(f"❌ Anthropic component test failed: {e}")
        print("💡 Check your Anthropic API key and component configuration")
        return False


def create_echo_component():
    """Create echo component configuration."""
    component_dir = Path("components")
    component_dir.mkdir(exist_ok=True)

    echo_config = """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: echo
spec:
  type: conversation.echo
  version: v1
  metadata: []
"""

    echo_file = component_dir / "echo-conversation.yaml"
    with open(echo_file, "w") as f:
        f.write(echo_config)

    print(f"✅ Created echo component: {echo_file}")


def create_openai_component():
    """Create OpenAI component configuration."""
    component_dir = Path("components")
    component_dir.mkdir(exist_ok=True)

    openai_config = """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  version: v1
  metadata:
  - name: apiKey
    value: "{openai_api_key}"
  - name: model
    value: "gpt-3.5-turbo"
""".format(openai_api_key=os.getenv("OPENAI_API_KEY", "your_openai_api_key_here"))

    openai_file = component_dir / "openai-conversation.yaml"
    with open(openai_file, "w") as f:
        f.write(openai_config)

    print(f"✅ Created OpenAI component: {openai_file}")


def create_anthropic_component():
    """Create Anthropic component configuration."""
    component_dir = Path("components")
    component_dir.mkdir(exist_ok=True)

    anthropic_config = """apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: anthropic
spec:
  type: conversation.anthropic
  version: v1
  metadata:
  - name: apiKey
    value: "{anthropic_api_key}"
  - name: model
    value: "claude-3-haiku-20240307"
""".format(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
    )

    anthropic_file = component_dir / "anthropic-conversation.yaml"
    with open(anthropic_file, "w") as f:
        f.write(anthropic_config)

    print(f"✅ Created Anthropic component: {anthropic_file}")


def show_component_configurations():
    """Show available component configurations."""
    print("\n📋 Component Configuration Examples")
    print("=" * 50)

    print("\n🔊 Echo Component (echo-conversation.yaml):")
    print("- No API key required")
    print("- Useful for testing")
    print("- Simply echoes back the input")

    print("\n🤖 OpenAI Component (openai-conversation.yaml):")
    print("- Requires OPENAI_API_KEY environment variable")
    print("- Uses gpt-3.5-turbo model")
    print("- Supports streaming and non-streaming")

    print("\n🧠 Anthropic Component (anthropic-conversation.yaml):")
    print("- Requires ANTHROPIC_API_KEY environment variable")
    print("- Uses claude-3-haiku-20240307 model")
    print("- Supports streaming and non-streaming")

    print("\n💡 Auto-create components with:")
    print("   python comprehensive_dapr_llm_testing.py --create-components")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Comprehensive Dapr LLM Testing")
    parser.add_argument(
        "--component",
        choices=["echo", "openai", "anthropic"],
        help="Test specific component",
    )
    parser.add_argument(
        "--show-config", action="store_true", help="Show component configurations"
    )
    parser.add_argument(
        "--create-components",
        action="store_true",
        help="Create component configuration files",
    )

    args = parser.parse_args()

    print("🧪 Comprehensive Dapr LLM Testing Suite")
    print("=" * 50)

    if args.show_config:
        show_component_configurations()
        return

    if args.create_components:
        print("📁 Creating component configurations...")
        create_echo_component()
        create_openai_component()
        create_anthropic_component()
        print("\n✅ All components created!")
        return

    if not check_prerequisites():
        sys.exit(1)

    # Test specific component or all
    results = {}

    if args.component == "echo" or not args.component:
        results["echo"] = test_echo_component()

    if args.component == "openai" or not args.component:
        results["openai"] = test_openai_component()

    if args.component == "anthropic" or not args.component:
        results["anthropic"] = test_anthropic_component()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{component.upper()}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check configuration and API keys")


if __name__ == "__main__":
    main()
