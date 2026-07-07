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
import os
from typing import Any

from dapr_agents import AgentRunner
from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.workflow.utils.core import wait_for_shutdown

from dapr_agents.ext.drasi import DrasiChangeEvent, drasi_trigger

from agent import make_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_PUBSUB_COMPONENT = os.getenv("AGENT_PUBSUB_COMPONENT", "agent-pubsub")


def make_task(event: DrasiChangeEvent, ctx: Any) -> TriggerAction:
    return TriggerAction(
        task=(
            f"You are an inventory agent that creates purchase orders, calculating the order quantity dynamically.\n"
            f"Create an inventory order for this '{event.payload.source.queryId}' event.\n"
            f"Use the following data: {event.payload.after}.\n\n"
            "Respond with exactly the following format, and nothing else:\n\n"
            "Product ID: <productId>\n"
            "Product Name: <productName>\n"
            "Product Description: <productDescription>\n"
            "Order Quantity: <quantity>\n\n"
            "Rules:\n"
            "- Output exactly these 4 lines, in this exact order.\n"
            "- Do not add, remove, rename, or reorder any fields.\n"
            "- Do not include any explanation, preamble, or extra text.\n"
            "- Do not wrap the output in code blocks or markdown formatting."
            "- Replace each <placeholder> with the actual value only — do not include the angle brackets."
        )
    )


async def main() -> None:
    agent = make_agent()

    # Register Drasi query subscriptions
    drasi_trigger(
        agent,
        query_id="critical-stock-event-query",
        pubsub=AGENT_PUBSUB_COMPONENT,
        task_mapper=make_task,
        operations="i",
    )
    drasi_trigger(
        agent,
        query_id="low-stock-event-query",
        pubsub=AGENT_PUBSUB_COMPONENT,
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
