from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("TestServer")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."


@mcp.tool()
async def jump(distance: str) -> str:
    """Simulate a jump of a given distance."""
    return f"I jumped the following distance: {distance}"


# When run directly, serve tools over STDIO
if __name__ == "__main__":
    mcp.run("stdio")
