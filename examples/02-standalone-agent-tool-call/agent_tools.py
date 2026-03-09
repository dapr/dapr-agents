#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
