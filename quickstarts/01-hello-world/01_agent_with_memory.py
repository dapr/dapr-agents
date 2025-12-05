import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import tool, Agent

from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.memory import ConversationDaprStateMemory


@tool
async def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"


async def main() -> None:
    # Create an agent with memory and tools
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func],
        # Configure this agent to use Dapr Conversation API.
        llm=DaprChatClient(component_name="openai"),
        # Configure the agent to use Dapr State Store for conversation history.
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversation-statestore",
                session_id="01-agent-with-memory",
            )
        ),
    )
    try:
        response = await weather_agent.run(
            "Hi, this is John, what's the weather in London?"
        )
        print(f"Agent: {response}")

        # Second interaction - agent remembers the name from the first interaction
        response = await weather_agent.run("What's my name?")
        print(f"Agent: {response}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
