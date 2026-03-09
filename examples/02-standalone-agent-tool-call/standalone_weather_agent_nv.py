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
from agent_tools import tools
from dapr_agents import Agent, NVIDIAChatClient
from dotenv import load_dotenv

load_dotenv()

llm = NVIDIAChatClient(model="meta/llama-3.1-8b-instruct")

AIAgent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Always answer the user's main weather question directly and clearly.",
        "If you perform any additional actions (like jumping), summarize those actions and their results.",
        "At the end, provide a concise summary that combines the weather information for all requested locations and any other actions you performed.",
    ],
    llm=llm,
    tools=tools,
)


# Wrap your async call
async def main():
    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")


if __name__ == "__main__":
    asyncio.run(main())
