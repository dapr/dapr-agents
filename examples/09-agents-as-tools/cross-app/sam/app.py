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
Cross-app agents-as-tools: Sam service.

Sam runs as a standalone Dapr app (app-id: SamApp).
By registering in the shared registry, Sam is automatically discoverable
as a callable tool by any other agent on the same registry.
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sam-app")

logging.getLogger("durabletask-client").propagate = False
logging.getLogger("durabletask-worker").propagate = False
logging.getLogger("WorkflowRuntime").propagate = False


def main() -> None:
    llm = DaprChatClient(component_name="llm-provider")

    registry = AgentRegistryConfig(
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agent-registry")
        ),
        team_name="fellowship",
    )

    sam = DurableAgent(
        name="sam",
        role="Sam Gamgee — trusted companion and logistics expert",
        goal=(
            "Provide practical, grounded support: manage provisions, "
            "navigate terrain, and keep spirits high."
        ),
        instructions=[
            "Answer questions about supplies and route logistics clearly.",
            "Keep replies concise and actionable.",
        ],
        llm=llm,
        registry=registry,
        pubsub=AgentPubSubConfig(
            pubsub_name=os.getenv("PUBSUB_NAME", "agent-pubsub"),
            agent_topic=os.getenv("SAM_TOPIC", "fellowship.sam.requests"),
            broadcast_topic=os.getenv("BROADCAST_TOPIC", "fellowship.broadcast"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(
                store_name=os.getenv("WORKFLOW_STATE_STORE", "agent-workflow"),
                key_prefix="sam:",
            )
        ),
    )

    runner = AgentRunner()
    try:
        runner.serve(sam, port=int(os.getenv("APP_PORT", "8002")))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
