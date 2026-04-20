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

"""
Smoke test for the AgentExecutorBase abstraction (RFC dapr/dapr-agents#569,
Linear AI-505).

Runs a DurableAgent backed by :class:`EchoAgentExecutor` — a trivial,
zero-dependency implementation of :class:`AgentExecutorBase` that echoes
the prompt back through the full event stream
(``text_delta`` -> ``message`` -> ``session`` -> ``complete``).

The purpose is to exercise the executor branch in
``DurableAgent.agent_workflow`` end-to-end against a real Dapr sidecar:

* The workflow reaches the ``elif self.executor is not None`` branch.
* The ``run_executor`` activity drives the async event stream.
* Dapr state picks up the ``session_id`` + user/assistant messages.

No LLM provider or API key is required.
"""

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
        print(
            "Inspect state: `redis-cli --scan --pattern 'echo*'` then "
            "`redis-cli GET <key>` to see the AgentWorkflowEntry with the "
            "populated session_id and messages.",
            flush=True,
        )
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
