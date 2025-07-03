import asyncio
import sys
from dotenv import load_dotenv

from dapr_agents import Agent
from dapr_agents.tool.mcp import MCPClient

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

    # Get available tools from the MCP instance
    tools = client.get_all_tools()
    print("🔧 Available tools:", [t.name for t in tools])

    # Create the Weather Agent using MCP tools
    weather_agent = Agent(
        name="Stevie",
        role="Weather Assistant",
        goal="Help humans get weather and location info using MCP tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch or simulate weather data.",
            "You may sometimes jump after answering the weather question.",
        ],
        tools=tools,
    )

    # Run a sample query
    result = await weather_agent.run("What is the weather in New York?")
    print(result)

    # Clean up resources
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
