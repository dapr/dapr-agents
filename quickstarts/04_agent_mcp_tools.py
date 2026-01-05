import asyncio
import sys

from dapr_agents.llm import DaprChatClient

from dapr_agents import Agent
from dapr_agents.tool.mcp import MCPClient


async def _load_mcp_tools() -> list:
    client = MCPClient()
    try:
        await client.connect_stdio(
            server_name="local",
            command=sys.executable,
            args=["mcp_tools.py"],
        )
        return client.get_all_tools()
    finally:
        await client.close()


def main() -> None:
    tools = asyncio.run(_load_mcp_tools())

    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=[
            "Provide concise, friendly weather info. Use MCP tools as needed."
        ],
        tools=tools,
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
