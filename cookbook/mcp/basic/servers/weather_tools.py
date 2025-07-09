from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("WeatherServer")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."


@mcp.tool()
async def get_humidity(location: str) -> str:
    """Get humidity information for a specific location."""
    humidity = random.randint(30, 90)
    return f"{location}: {humidity}% humidity."
