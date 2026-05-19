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

from dapr_agents import DurableAgent, EchoAgentExecutor
from dapr_agents.workflow.runners import AgentRunner


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    agent = DurableAgent(
        role="Echo Assistant",
        name="Echo",
        goal="Repeat user input through the AgentExecutorBase event stream.",
        executor=EchoAgentExecutor(chunk_size=8),
    )

    runner = AgentRunner()
    try:
        prompt = "hello, agent executor"
        logger.info("Triggering Echo agent with prompt: %r", prompt)

        result = await runner.run(agent, payload={"task": prompt})

        print("\n=== Final Result ===", flush=True)
        print(result, flush=True)
        print("====================\n", flush=True)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
