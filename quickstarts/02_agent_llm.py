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

from dapr_agents.llm import DaprChatClient

from dapr_agents import Agent


def main() -> None:
    # Simple agent: LLM, no tools, no memory
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Provide concise, friendly weather info."],
        llm=DaprChatClient(component_name="llm-provider"),
    )

    try:
        response = asyncio.run(
            weather_agent.run("What's a quick weather update for London right now?")
        )
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
