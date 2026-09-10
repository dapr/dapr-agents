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

from __future__ import annotations

import asyncio
import logging
from typing import Any

from dapr_agents import AgentRunner
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.workflow.utils.core import wait_for_shutdown

from dapr_agents.ext.drasi import DrasiChangeEvent, drasi_trigger

from agent import make_agent

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def make_task(event: DrasiChangeEvent, ctx: Any) -> TriggerAction:
    return TriggerAction(
        task=(
            f"You are an inventory agent that creates purchase orders in response to stock events, "
            "calculating the order quantity dynamically.\n\n"
            f"## Event Data\n"
            f"{event.payload.after.model_dump_json() if event.payload.after else 'N/A'}\n\n"
            f"## Response Format\n"
            "Product ID: <productId from Event Data>\n"
            "Product Name: <productName from Event Data>\n"
            "Product Description: <productDescription from Event Data>\n"
            "Order Quantity: <quantity to be calculated>\n\n"
            "## Rules\n"
            "- Respond EXACTLY in the given response format, and nothing else.\n"
            "- Do NOT add, remove, rename, or reorder any fields.\n"
            "- Do NOT include any explanation, preamble, or extra text.\n"
            "- Do NOT wrap the output in code blocks (no ``` fences) or markdown formatting.\n"
            "- Replace each <placeholder> with the actual value only — do NOT include the angle brackets."
        )
    )


async def main() -> None:
    agent = make_agent()

    # Register Drasi query subscriptions
    drasi_trigger(
        agent,
        query_id="critical-stock-event-query",
        task_mapper=make_task,
        operations="i",
    )
    drasi_trigger(
        agent,
        query_id="low-stock-event-query",
        task_mapper=make_task,
        operations="i",
    )

    runner = AgentRunner()
    try:
        runner.subscribe(agent)
        await wait_for_shutdown()
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
