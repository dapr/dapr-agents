from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

import dapr.ext.workflow as wf
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.orchestrators.random import RandomOrchestrator
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.workflow.utils.core import wait_for_shutdown

# -----------------------------------------------------------------------------
# Boot
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fellowship.orchestrator.random.app")


async def main() -> None:
    """
    Fellowship Orchestrator (Random) application.

    This service hosts the RandomOrchestrator on a Dapr Workflow runtime.
    It subscribes to the orchestrator topic for TriggerAction messages and
    coordinates registered agents (e.g., Frodo, Sam) by selecting one at random
    each turn and routing replies back into the workflow via external events.
    """

    # -------------------------------------------------------------------------
    # Config (env-overridable)
    # -------------------------------------------------------------------------
    orchestrator_name = os.getenv("ORCHESTRATOR_NAME", "FellowshipRandom")
    team_name = os.getenv("TEAM_NAME", "fellowship")

    # Pub/Sub topics: orchestrator listens on agent_topic for TriggerAction
    pubsub_name = os.getenv("PUBSUB_NAME", "messagepubsub")
    orchestrator_topic = os.getenv(
        "ORCHESTRATOR_TOPIC", "fellowship.orchestrator.random.requests"
    )
    broadcast_topic = os.getenv("BROADCAST_TOPIC", "fellowship.broadcast")

    # (Optional) state & registry stores (by name, as configured in Dapr components)
    workflow_state_store_name = os.getenv("WORKFLOW_STATE_STORE", "workflowstatestore")
    registry_store_name = os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")

    # Orchestrator behavior
    max_iterations = int(os.getenv("MAX_ITERATIONS", "8"))
    timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "45"))

    # -------------------------------------------------------------------------
    # Pub/Sub, State, Registry wiring
    # -------------------------------------------------------------------------
    pubsub = AgentPubSubConfig(
        pubsub_name=pubsub_name,
        agent_topic=orchestrator_topic,  # <-- RandomOrchestrator subscribes here
        broadcast_topic=broadcast_topic,  # <-- Optional (fanout to agents)
    )

    # Orchestrators often don’t persist workflow-local state; still allow it
    # so you can extend later (metrics, audit, etc).
    state = AgentStateConfig(
        store=StateStoreService(
            store_name=workflow_state_store_name, key_prefix="fellowship.random:"
        ),
    )

    registry = AgentRegistryConfig(
        store=StateStoreService(store_name=registry_store_name),
        team_name=team_name,
    )

    execution = AgentExecutionConfig(max_iterations=max_iterations)

    # -------------------------------------------------------------------------
    # Orchestrator instance
    # -------------------------------------------------------------------------
    orchestrator = RandomOrchestrator(
        name=orchestrator_name,
        pubsub=pubsub,
        state=state,
        registry=registry,
        agent_metadata={"legend": "One orchestrator to guide them all."},
        execution=execution,
        timeout_seconds=timeout_seconds,
        runtime=wf.WorkflowRuntime(),  # you can inject your own if needed
    )

    # Start workflow runtime + register workflows/activities
    orchestrator.start()

    # -------------------------------------------------------------------------
    # HTTP runner (exposes workflow endpoints for Dapr to call)
    # -------------------------------------------------------------------------
    runner = AgentRunner()
    try:
        runner.register_routes(orchestrator)
        await wait_for_shutdown()
    finally:
        runner.shutdown()
        orchestrator.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
