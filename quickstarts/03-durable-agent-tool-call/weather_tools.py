from dapr_agents import tool
from pydantic import BaseModel, Field


class GetWeatherSchema(BaseModel):
    location: str = Field(description="location to get weather for")


@tool(args_model=GetWeatherSchema)
def get_weather(location: str) -> str:
    """Get weather information based on location."""
    import random

    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."


class JumpSchema(BaseModel):
    distance: str = Field(description="Distance for agent to jump")


@tool(args_model=JumpSchema)
def jump(distance: str) -> str:
    """Jump a specific distance."""
    return f"I jumped the following distance {distance}"


tools = [get_weather, jump]
