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
import logging

from dapr_agents import AgentRunner, DurableAgent

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    agent = DurableAgent(
        name="WeatherAgent",
        role="Weather assistant",
        goal="Answer weather questions using MCP tools.",
        instructions=["Use the available tools to look up weather information."],
    )

    async with AgentRunner() as runner:
        await runner.run(
            agent,
            payload={"task": "What is the weather in Seattle?"},
            wait=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
