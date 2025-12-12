import asyncio

from dapr_agents.llm import DaprChatClient

from dapr_agents import Agent
from function_tools import weather_func


def main() -> None:
    # Simple agent: LLM + tools, but no memory
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Provide concise, friendly weather info."],
        tools=[weather_func],
        llm=DaprChatClient(component_name="llm-provider"),
    )

    try:
        response = asyncio.run(
            weather_agent.run("What's a quick weather update for London right now?")
        )
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
