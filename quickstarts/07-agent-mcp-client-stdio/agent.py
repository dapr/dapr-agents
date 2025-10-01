import asyncio
import sys
from dotenv import load_dotenv

from dapr_agents import Agent
from dapr_agents.tool.mcp import MCPClient
from dapr_agents.types import AssistantMessage
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()


async def main():
    # Create the MCP client
    client = MCPClient()

    # Connect to MCP server using STDIO transport
    await client.connect_stdio(
        server_name="local",
        command=sys.executable,  # Use the current Python interpreter
        args=["tools.py"],  # Run tools.py directly
    )

    try:
        tools = client.get_all_tools()
        print("🔧 Available tools:", [t.name for t in tools])

        # Create the Weather Agent using MCP tools
        weather_agent = Agent(
            name="Stevie",
            role="Weather Assistant",
            goal="Help humans get weather and location info using MCP tools.",
            llm=OpenAIChatClient(model="gpt-4o-mini"),
            instructions=[
                "Respond clearly and helpfully to weather-related questions.",
                "Use tools when appropriate to fetch or simulate weather data.",
                "You may sometimes jump after answering the weather question.",
            ],
            tools=tools,
        )

        # Run a sample query
        result: AssistantMessage = await weather_agent.run(
            "What is the weather in New York?"
        )
        print(result.content)

    finally:
        try:
            await client.close()
        except RuntimeError as e:
            if "Attempted to exit cancel scope" not in str(e):
                raise


if __name__ == "__main__":
    asyncio.run(main())
