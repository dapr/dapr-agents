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

from agent_tools import tools
from dotenv import load_dotenv

from dapr_agents import DurableAgent, HFHubChatClient
from dapr_agents.workflow.runners import AgentRunner


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Initialize the HuggingFaceChatClient with the desired model
    llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")

    # Instantiate your agent (no .as_service())
    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        llm=llm,
        tools=tools,
    )

    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        prompt = "What's the current weather in Boston, MA?"

        # Run the workflow and wait for completion
        result = await runner.run(
            weather_agent,
            payload={"task": prompt},
        )

        print(f"\n✅ Final Result:\n{result}\n", flush=True)

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        raise
    finally:
        # Then shut down runner (unwire/close clients)
        runner.shutdown(weather_agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
