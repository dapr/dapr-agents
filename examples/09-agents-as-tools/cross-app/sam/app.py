"""
Cross-app agents-as-tools: Sam service.

Sam runs as a standalone Dapr app (app-id: SamApp).
Setting ``is_tool=True`` publishes the ``is_tool`` flag to the shared registry,
so that any agent sharing that registry can discover and call Sam as a tool automatically.

Sam can also be targeted explicitly via ``agent_to_tool`` without a registry by passing ``target_app_id="SamApp"``.
Alternatively, as long as Sam and Frodo are registered under the same team,
then Frodo will see that Sam is enabled as a tool and can call it automatically.
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
        is_tool=True,  # advertise this agent as callable by others
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
