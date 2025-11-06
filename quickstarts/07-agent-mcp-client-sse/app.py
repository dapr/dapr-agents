import asyncio
import logging
from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.tool.mcp import MCPClient


async def main():
    try:
        # Load MCP tools from server (stdio or sse)
        client = MCPClient()
        await client.connect_sse("local", url="http://localhost:8000/sse")

        # Convert MCP tools to AgentTool list
        tools = client.get_all_tools()

        # Create the Weather Agent using those tools
        weather_agent = DurableAgent(
            role="Weather Assistant",
            name="Stevie",
            goal="Help humans get weather and location info using smart tools.",
            instructions=[
                "Respond clearly and helpfully to weather-related questions.",
                "Use tools when appropriate to fetch or simulate weather data.",
                "You may sometimes jump after answering the weather question.",
            ],
            tools=tools,
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
        ).as_service(port=8001)

        # Start the FastAPI agent service
        await weather_agent.start()

    except Exception as e:
        logging.exception("Error starting weather agent service", exc_info=e)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
