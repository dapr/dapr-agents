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

import asyncio
import random

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
