import asyncio
import random
import re

from dapr_agents import tool


@tool
async def weather_func(location: str) -> str:
    """Get weather information for a specific location."""
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."


@tool
async def slow_weather_func(location: str) -> str:
    """Get weather information for a specific location with a simulated delay."""
    await asyncio.sleep(5)
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."
