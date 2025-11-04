from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    AgentExecutionConfig,
)
from dapr_agents.agents.orchestrators.roundrobin import RoundRobinOrchestrator
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
logger = logging.getLogger("fellowship.orchestrator.roundrobin.app")


async def main() -> None:
    """
    Fellowship Orchestrator (RoundRobin).

    Hosts RoundRobinOrchestrator on a Dapr Workflow runtime.
    Subscribes to the orchestrator topic for TriggerAction messages and
    coordinates agents by selecting them in round-robin order each turn.
    """

    # -------------------------------------------------------------------------
    # Config (env-overridable)
    # -------------------------------------------------------------------------
    orchestrator_name = os.getenv("ORCHESTRATOR_NAME", "FellowshipRoundRobin")
    team_name = os.getenv("TEAM_NAME", "fellowship")

    # Pub/Sub topics: orchestrator listens on this topic for TriggerAction
    pubsub_name = os.getenv("PUBSUB_NAME", "messagepubsub")
    orchestrator_topic = os.getenv(
        "ORCHESTRATOR_TOPIC", "fellowship.orchestrator.roundrobin.requests"
    )
    broadcast_topic = os.getenv("BROADCAST_TOPIC", "fellowship.broadcast")

    # Dapr state components (by name)
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
        agent_topic=orchestrator_topic,  # RoundRobinOrchestrator subscribes here
        broadcast_topic=broadcast_topic,  # optional fan-out to agents
    )

    state = AgentStateConfig(
        store=StateStoreService(
            store_name=workflow_state_store_name, key_prefix="fellowship.roundrobin:"
        ),
    )

    registry = AgentRegistryConfig(
        store=StateStoreService(store_name=registry_store_name),
        team_name=team_name,
    )

    execution = AgentExecutionConfig(max_iterations=3)

    # -------------------------------------------------------------------------
    # Orchestrator instance
    # -------------------------------------------------------------------------
    # Recommended: let the orchestrator OWN the runtime (don’t pass runtime=...).
    orchestrator = RoundRobinOrchestrator(
        name=orchestrator_name,
        pubsub=pubsub,
        state=state,
        registry=registry,
        agent_metadata={"pattern": "Round-robin selection of agents."},
        execution=execution,
        timeout_seconds=timeout_seconds,
    )

    orchestrator.start()  # registers and starts owned runtime

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
