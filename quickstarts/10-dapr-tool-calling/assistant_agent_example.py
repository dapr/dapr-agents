#!/usr/bin/env python3

"""
AssistantAgent Tool Calling Example

This example demonstrates using AssistantAgent (workflow-based) for tool calling
with Dapr conversation components. AssistantAgent provides:

- Persistent memory and conversation history
- Automatic tool calling iteration
- Workflow-based execution with state management
- Multi-agent communication capabilities

Usage:
    python assistant_agent_example.py
    python assistant_agent_example.py --provider openai
    python assistant_agent_example.py --provider anthropic
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from dapr_agents import tool, AssistantAgent
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.tool.base import AgentTool


# Load environment variables from .env file at repo root
def load_env_file():
    """Load environment variables from .env file at the root of the repo."""
    # Get the repo root (two levels up from this file)
    repo_root = Path(__file__).parent.parent.parent
    env_file = repo_root / ".env"

    if env_file.exists():
        print(f"📄 Loading environment variables from: {env_file}")
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            print("✅ Environment variables loaded successfully")
        except ImportError:
            print(
                "⚠️  python-dotenv not installed. Install with: pip install python-dotenv"
            )
            print("⚠️  Falling back to system environment variables")
        except Exception as e:
            print(f"⚠️  Error loading .env file: {e}")
            print("⚠️  Falling back to system environment variables")
    else:
        print(f"📄 No .env file found at: {env_file}")
        print("⚠️  Using system environment variables")


# Define tool output models for better type safety
class WeatherInfo(BaseModel):
    location: str = Field(description="Location name")
    temperature: str = Field(description="Temperature")
    conditions: str = Field(description="Weather conditions")
    wind: str = Field(description="Wind information")


class CalculationResult(BaseModel):
    expression: str = Field(description="Mathematical expression")
    result: str = Field(description="Calculation result")


class TimeZoneInfo(BaseModel):
    location: str = Field(description="Location name")
    timezone: str = Field(description="Time zone information")


# Define function-based tools using @tool decorator
@tool
def get_weather(location: str) -> WeatherInfo:
    """Get current weather conditions for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'

    Returns:
        Current weather information
    """
    return WeatherInfo(
        location=location, temperature="72°F", conditions="sunny", wind="light breeze"
    )


@tool
def calculate(expression: str) -> CalculationResult:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression)
        return CalculationResult(expression=expression, result=str(result))
    except Exception as e:
        return CalculationResult(expression=expression, result=f"Error: {str(e)}")


@tool
def get_time_zone(location: str) -> TimeZoneInfo:
    """Get the current time zone information for a location.

    Args:
        location: The city and country, e.g. 'Tokyo, Japan'

    Returns:
        Time zone information
    """
    # Simple mock time zone data
    time_zones = {
        "tokyo": "JST (UTC+9)",
        "london": "GMT (UTC+0)",
        "new york": "EST (UTC-5)",
        "san francisco": "PST (UTC-8)",
        "sydney": "AEDT (UTC+11)",
        "paris": "CET (UTC+1)",
    }

    location_key = location.lower().split(",")[0].strip()
    timezone = time_zones.get(location_key, "Unknown timezone")

    return TimeZoneInfo(location=location, timezone=timezone)


# Define class-based tools inheriting from AgentTool
class CurrencyConverter(AgentTool):
    """Convert currency from one type to another."""

    def __init__(self):
        super().__init__(
            name="convert_currency",
            description="Convert an amount from one currency to another",
        )

    def _run(self, amount: float, from_currency: str, to_currency: str) -> str:
        """Execute the currency conversion.

        Args:
            amount: The amount to convert
            from_currency: The source currency code (e.g., 'USD')
            to_currency: The target currency code (e.g., 'EUR')
        """
        # Simple mock exchange rates
        rates = {
            ("USD", "EUR"): 0.85,
            ("EUR", "USD"): 1.18,
            ("USD", "GBP"): 0.73,
            ("GBP", "USD"): 1.37,
            ("USD", "JPY"): 110.0,
            ("JPY", "USD"): 0.009,
        }

        rate = rates.get((from_currency.upper(), to_currency.upper()))
        if rate:
            converted_amount = amount * rate
            return f"{amount} {from_currency.upper()} = {converted_amount:.2f} {to_currency.upper()}"
        else:
            return f"Exchange rate not available for {from_currency} to {to_currency}"


class TaskManager(AgentTool):
    """Manage a simple task list."""

    def __init__(self):
        super().__init__(
            name="manage_tasks",
            description="Add, list, or complete tasks in a simple task list",
        )
        self.tasks = []

    def _run(self, action: str, task: str = None) -> str:
        """Execute task management operations.

        Args:
            action: The action to perform ('add', 'list', 'complete')
            task: The task description (required for 'add' and 'complete')
        """
        if action == "add" and task:
            self.tasks.append(
                {"id": len(self.tasks) + 1, "task": task, "completed": False}
            )
            return f"Added task: {task}"
        elif action == "list":
            if not self.tasks:
                return "No tasks in the list"
            task_list = []
            for t in self.tasks:
                status = "✓" if t["completed"] else "○"
                task_list.append(f"{status} {t['id']}. {t['task']}")
            return "Tasks:\n" + "\n".join(task_list)
        elif action == "complete" and task:
            for t in self.tasks:
                if t["task"].lower() == task.lower():
                    t["completed"] = True
                    return f"Completed task: {task}"
            return f"Task not found: {task}"
        else:
            return "Invalid action. Use 'add', 'list', or 'complete'"


def check_provider_requirements(provider: str):
    """Check if the required API keys are available for the provider."""

    # Echo provider is not supported for AssistantAgent
    if provider == "echo":
        print("❌ Echo provider is not supported for AssistantAgent!")
        print("   Echo doesn't support tool calling - use 'openai' or 'anthropic'")
        print("   For testing without API keys, use Simple Agent approach instead")
        return False

    requirements = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_AI_API_KEY",
        "googleai": "GOOGLE_AI_API_KEY",
    }

    if provider in requirements:
        api_key = os.getenv(requirements[provider])
        if not api_key:
            print(
                f"❌ Error: {requirements[provider]} not found in environment variables or .env file"
            )
            print(f"   Provider '{provider}' requires an API key for tool calling")
            print("   Set the environment variable or add to .env file")
            return False
        else:
            print(f"✅ API key found for provider '{provider}'")
            return True
    else:
        print(f"⚠️  Unknown provider '{provider}' - proceeding anyway")
        return True


async def run_assistant_agent_example(
    provider: str = "openai",
    use_class_tools: bool = False,
    session_id: str = "demo-session",
):
    """Run the AssistantAgent tool calling example."""

    print(f"🤖 Starting AssistantAgent with provider: {provider}")
    print(f"🔧 Tool type: {'Class-based' if use_class_tools else 'Function-based'}")
    print(f"💾 Session ID: {session_id}")
    print("=" * 60)

    # Check provider requirements
    if not check_provider_requirements(provider):
        print("\n❌ Provider requirements not met. Exiting...")
        return

    # Choose tools based on the flag
    if use_class_tools:
        # Use class-based tools
        tools = [CurrencyConverter(), TaskManager()]
    else:
        # Use function-based tools
        tools = [get_weather, calculate, get_time_zone]

    try:
        # Initialize AssistantAgent
        assistant = AssistantAgent(
            name="ToolAssistant",
            role="Helpful Assistant",
            goal="Help users with various tasks using available tools",
            instructions=[
                "Use tools when appropriate to help users",
                "Provide clear and helpful responses",
                "Remember conversation history",
                "Be friendly and professional",
            ],
            tools=tools,
            llm=DaprChatClient(component_name=provider),
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="assistant_workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="agents_registry",
            memory=ConversationDaprStateMemory(
                store_name="conversationstore", session_id=session_id
            ),
            max_iterations=5,  # Allow multiple tool calling iterations
        )

        # Start the assistant as a service
        print("🚀 Starting AssistantAgent service...")
        assistant.as_service(port=8002)
        await assistant.start()

        print("✅ AssistantAgent is running!")
        print("\n🎯 The agent will automatically:")
        print("   • Process tool calls in workflows")
        print("   • Remember conversation history")
        print("   • Iterate on complex tasks")
        print("   • Manage state persistently")

        print("\n📝 To interact with the agent:")
        print("   • Send messages via Dapr workflow triggers")
        print("   • Use the agent's REST API endpoints")
        print("   • Integrate with other Dapr services")

        print("\n🔗 Available tools:")
        for t in tools:
            if hasattr(t, "name"):
                print(f"   • {t.name}: {t.description}")
            else:
                print(
                    f"   • {t.__name__}: {t.__doc__.split('.')[0] if t.__doc__ else 'No description'}"
                )

        print("\n⏳ Service is running... Press Ctrl+C to stop")

        # Keep the service running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping AssistantAgent service...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def run_interactive_demo(
    provider: str = "echo", session_id: str = "demo-session"
):
    """Run an interactive demo showing AssistantAgent capabilities."""

    print("🎮 Interactive AssistantAgent Demo")
    print("=" * 40)

    # Simple tools for demo
    tools = [get_weather, calculate]

    try:
        # Initialize AssistantAgent
        assistant = AssistantAgent(
            name="DemoAssistant",
            role="Demo Assistant",
            goal="Demonstrate tool calling capabilities",
            instructions=[
                "Help users with weather and calculations",
                "Use tools when needed",
                "Be helpful and clear",
            ],
            tools=tools,
            llm=DaprChatClient(component_name=provider),
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="demo_workflow_state",
            agents_registry_store_name="registrystatestore",
            agents_registry_key="agents_registry",
            memory=ConversationDaprStateMemory(
                store_name="conversationstore", session_id=session_id
            ),
        )

        print("🚀 Starting demo agent...")
        assistant.as_service(port=8003)
        await assistant.start()

        print("✅ Demo agent is ready!")
        print("\nThis demonstrates AssistantAgent with:")
        print("• Persistent memory across conversations")
        print("• Automatic tool calling workflows")
        print("• State management via Dapr")
        print("• Multi-iteration task processing")

        print("\n⏳ Demo service running... Press Ctrl+C to stop")

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping demo service...")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="AssistantAgent Tool Calling Example")
    parser.add_argument(
        "--provider",
        default="openai",
        help="Dapr conversation provider (openai, anthropic, gemini) - echo not supported",
    )
    parser.add_argument(
        "--class-tools",
        action="store_true",
        help="Use class-based tools instead of function-based tools",
    )
    parser.add_argument(
        "--session-id",
        default="demo-session",
        help="Session ID for conversation memory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo mode",
    )

    args = parser.parse_args()

    tool_type = "Class-based" if args.class_tools else "Function-based"
    mode = "Demo" if args.demo else "Full"

    if args.class_tools:
        tools_desc = """
• CurrencyConverter - Convert amounts between currencies
• TaskManager - Manage a simple task list (add, list, complete)"""
    else:
        tools_desc = """
• get_weather(location) - Get weather for a location
• calculate(expression) - Perform mathematical calculations
• get_time_zone(location) - Get time zone information"""

    print(
        f"""
🤖 AssistantAgent Tool Calling Example

This example demonstrates workflow-based tool calling using AssistantAgent.
AssistantAgent provides advanced features like persistent memory, automatic
tool calling iteration, and workflow-based execution.

Provider: {args.provider}
Tool Type: {tool_type}
Mode: {mode}
Session ID: {args.session_id}

Available tools:{tools_desc}

Prerequisites:
• Dapr runtime with workflow support
• Redis for state storage and pub/sub
• Conversation components configured

Make sure to run: python tools/run_dapr_dev.py --components ./tests/components/local_dev
"""
    )

    # Load environment variables from .env file
    load_env_file()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the appropriate example
    if args.demo:
        asyncio.run(run_interactive_demo(args.provider, args.session_id))
    else:
        asyncio.run(
            run_assistant_agent_example(
                args.provider, args.class_tools, args.session_id
            )
        )


if __name__ == "__main__":
    main()
