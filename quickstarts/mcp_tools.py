from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("TestServer")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."


if __name__ == "__main__":
    mcp.run("stdio")
