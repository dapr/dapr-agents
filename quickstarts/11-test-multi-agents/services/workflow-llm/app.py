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
    AgentExecutionConfig,
)
from dapr_agents.agents.orchestrators.llm import LLMOrchestrator
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.workflow.utils.core import wait_for_shutdown


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("llm.orchestrator.app")


async def main() -> None:
    """
    LLM Orchestrator application.

    This service hosts the LLMOrchestrator on a Dapr Workflow runtime.
    It subscribes to the orchestrator topic for TriggerAction messages and
    coordinates registered agents by selecting the next agent/step in the workflow.
    """

    # -------------------------------------------------------------------------
    # Config (env-overridable)
    # -------------------------------------------------------------------------
    orchestrator_name = os.getenv("ORCHESTRATOR_NAME", "LLMOrchestrator")
    team_name = os.getenv("TEAM_NAME", "fellowship")

    # Pub/Sub topics: orchestrator listens on orchestrator_topic for TriggerAction
    pubsub_name = os.getenv("PUBSUB_NAME", "messagepubsub")
    orchestrator_topic = os.getenv("ORCHESTRATOR_TOPIC", "llm.orchestrator.requests")
    broadcast_topic = os.getenv("BROADCAST_TOPIC", "fellowship.broadcast")

    # (Optional) state & registry stores (by name, as configured in Dapr components)
    workflow_state_store_name = os.getenv("WORKFLOW_STATE_STORE", "workflowstatestore")
    registry_store_name = os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")

    # Orchestrator behavior
    max_iterations = int(os.getenv("MAX_ITERATIONS", "8"))
    timeout_seconds = int(os.getenv("TIMEOUT_SECONDS", "45"))

    # LLM Provider
    llm = OpenAIChatClient()

    # -------------------------------------------------------------------------
    # Pub/Sub, State, Registry wiring
    # -------------------------------------------------------------------------
    pubsub = AgentPubSubConfig(
        pubsub_name=pubsub_name,
        agent_topic=orchestrator_topic,
        broadcast_topic=broadcast_topic,
    )

    # Orchestrators often don't persist workflow-local state; still allow it
    # so you can extend later (metrics, audit, etc).
    # Schema automatically set to LLMWorkflowState by LLMOrchestrator
    state = AgentStateConfig(
        store=StateStoreService(
            store_name=workflow_state_store_name, key_prefix="llm.orchestrator:"
        ),
    )

    registry = AgentRegistryConfig(
        store=StateStoreService(store_name=registry_store_name),
        team_name=team_name,
    )

    execution = AgentExecutionConfig(max_iterations=max_iterations)

    # -------------------------------------------------------------------------
    # LLM Orchestrator instance
    # -------------------------------------------------------------------------
    orchestrator = LLMOrchestrator(
        name=orchestrator_name,
        llm=llm,
        pubsub=pubsub,
        state=state,
        registry=registry,
        execution=execution,
        agent_metadata={
            "type": "LLMOrchestrator",
            "description": "LLM-driven Orchestrator",
        },
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
