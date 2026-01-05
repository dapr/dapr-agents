import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import Agent
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.memory import ConversationDaprStateMemory
from function_tools import weather_func


async def main() -> None:
    # Create an agent with memory and tools
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[weather_func],
        # Configure this agent to use Dapr Conversation API.
        llm=DaprChatClient(component_name="llm-provider"),
        # Configure the agent to use Dapr State Store for conversation history.
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="conversation-statestore",
                session_id="03-agent-with-memory",
            )
        ),
    )
    try:
        response = await weather_agent.run(
            "I like warm and dry places. What is the weather in London now?"
        )
        print(f"Agent: {response}")

        # Second interaction - agent remembers the preference from the first interaction
        response = await weather_agent.run(
            "Given my preference, is London’s current weather a good match?"
        )
        print(f"Agent: {response}")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
