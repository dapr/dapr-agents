#!/usr/bin/env python3

"""
Simple Agent Tool Calling Example

This example demonstrates the simplest way to use tool calling with Dapr agents.
It follows the same pattern as quickstart 03 but uses the enhanced DaprChatClient.

Usage:
    python simple_agent_example.py
    python simple_agent_example.py --provider openai
"""

import argparse
import asyncio
from pathlib import Path
from dapr_agents import Agent, tool
from dapr_agents.llm.dapr import DaprChatClient
from dotenv import load_dotenv


# Load environment variables from .env file at repo root
def load_env_file():
    """Load environment variables from .env file at the root of the repo."""
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded env from: {env_file}")


# Define simple tools
@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a location."""
    return f"Weather in {location}: 72°F, sunny, light breeze"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Simple safe evaluation for basic math
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


# Create the simple agent
def create_weather_agent(provider: str = "echo"):
    """Create a simple weather agent with enhanced DaprChatClient."""

    # Configure the enhanced DaprChatClient with the provider
    llm_client = DaprChatClient(component_name=provider)

    # Create agent with tools and LLM client
    agent = Agent(
        name="WeatherBot",
        role="Weather and Math Assistant",
        goal="Help users with weather information and calculations",
        instructions=[
            "Get accurate weather information for any location",
            "Perform mathematical calculations when requested",
            "Be helpful and friendly in your responses",
        ],
        tools=[get_weather, calculate],
        llm=llm_client,
    )

    return agent


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple Agent Tool Calling Example")
    parser.add_argument(
        "--provider",
        default="echo",
        help="Dapr conversation provider (echo, openai, anthropic, etc.)",
    )

    args = parser.parse_args()

    print(
        f"""
🤖 Simple Agent Tool Calling

This example shows the simplest way to use tool calling with Dapr agents:
• Enhanced DaprChatClient for tool calling
• Simple Agent configuration
• Automatic tool execution
• Provider: {args.provider}

Make sure Dapr is running with your components configured!
"""
    )

    # Load environment variables
    load_env_file()

    # Create the agent
    agent = create_weather_agent(args.provider)

    # Simple tool calling examples
    print("🌤️  Example 1: Weather query")
    await agent.run("What's the weather like in San Francisco?")

    print("\n" + "=" * 50)
    print("🧮 Example 2: Math calculation")
    await agent.run("Calculate 15 * 8 + 7")

    print("\n" + "=" * 50)
    print("🔧 Example 3: Multiple tools")
    await agent.run("Get the weather in Tokyo and calculate 25 * 4")

    print("\n✅ Simple tool calling examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
