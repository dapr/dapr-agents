"""
Cross-app agents-as-tools: Frodo service.

Frodo runs as a standalone Dapr app (app-id: FrodoApp).
Sam lives in a separate Dapr app (SamApp).
Two equivalent ways to wire Sam as a tool are:

  Option A — Registry-based auto-discovery (recommended):
      Pass ``tools=["sam"]`` and share a registry.
      Frodo resolves Sam's ``app_id`` at workflow start.

  Option B — Explicit factory (no registry needed):
      Use ``agent_to_tool("sam", ..., target_app_id="SamApp")`` and
      pass the result in ``tools``.
      Alternatively, use the agent name string as the tool.
"""

from __future__ import annotations

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
from dapr_agents.tool.workflow import agent_to_tool
from dapr_agents.workflow.runners import AgentRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("frodo-app")
logging.getLogger("durabletask-client").propagate = False
logging.getLogger("durabletask-worker").propagate = False
logging.getLogger("WorkflowRuntime").propagate = False


# Dapr app-id of the Sam service
SAM_APP_ID = os.getenv("SAM_APP_ID", "SamApp")


def main() -> None:
    llm = DaprChatClient(component_name="llm-provider")

    registry = AgentRegistryConfig(
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agent-registry")
        ),
        team_name="fellowship",
    )

    # Option A: pass the agent name as a string; resolved from registry at runtime.
    # sam_agent_as_tool = "sam"

    # Option B: explicit cross-app factory — no registry dependency.
    # sam_agent_as_tool = agent_to_tool(
    #     "sam",
    #     description="Sam Gamgee. Goal: Provide logistics and supply support.",
    #     target_app_id=SAM_APP_ID,
    # )

    # Option C: don't set any tool on frodo agent, as sam is already registered as an agent as a tool.

    frodo = DurableAgent(
        name="frodo",
        role="Frodo Baggins — ring-bearer",
        goal=(
            "Complete the quest to Mount Doom. "
            "Delegate logistics and supply questions to Sam."
        ),
        instructions=[
            "Use the 'sam' tool whenever you need supply or route information.",
            "Stay focused on the overall mission.",
        ],
        llm=llm,
        registry=registry,
        # Note: tools can be a mix of strings (registry lookup) and explicit AgentTool instances.
        # If using registry-based discovery, Sam will be auto-registered as a tool since is_tool=True in Sam's config.
        # tools=[sam_agent_as_tool],
        pubsub=AgentPubSubConfig(
            pubsub_name=os.getenv("PUBSUB_NAME", "agent-pubsub"),
            agent_topic=os.getenv("FRODO_TOPIC", "fellowship.frodo.requests"),
            broadcast_topic=os.getenv("BROADCAST_TOPIC", "fellowship.broadcast"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(
                store_name=os.getenv("WORKFLOW_STATE_STORE", "agent-workflow"),
                key_prefix="frodo:",
            )
        ),
    )

    runner = AgentRunner()
    try:
        runner.serve(frodo, port=int(os.getenv("APP_PORT", "8001")))
    finally:
        runner.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
