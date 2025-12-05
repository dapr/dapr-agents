import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner


# Notice this tool simulates a delay
@tool
async def my_weather_func() -> str:
    """Get current weather."""
    await asyncio.sleep(5)
    return "It's 72°F and sunny"


def main() -> None:
    # This agent is of type durable agent where the execution is durable
    weather_agent = DurableAgent(
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
                session_id="02-durable-agent",
            )
        ),
        # This is where the execution state is stored
        state=AgentStateConfig(
            store=StateStoreService(store_name="workflow-statestore"),
        ),
    )

    # This runner will run the agent and expose it on port 8001
    runner = AgentRunner()
    try:
        runner.serve(weather_agent, port=8001)
    finally:
        runner.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
