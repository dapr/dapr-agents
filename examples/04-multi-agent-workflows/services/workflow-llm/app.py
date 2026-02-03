from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

import dapr.ext.workflow as wf
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.orchestrators.llm import LLMOrchestrator
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("llm.orchestrator.app")


def main() -> None:
    orchestrator_name = os.getenv("ORCHESTRATOR_NAME", "LLMOrchestrator")
    team_name = os.getenv("TEAM_NAME", "fellowship")

    pubsub = AgentPubSubConfig(
        pubsub_name=os.getenv("PUBSUB_NAME", "messagepubsub"),
        agent_topic=os.getenv("ORCHESTRATOR_TOPIC", "llm.orchestrator.requests"),
        broadcast_topic=os.getenv("BROADCAST_TOPIC", "fellowship.broadcast"),
    )

    state = AgentStateConfig(
        store=StateStoreService(
            store_name=os.getenv("WORKFLOW_STATE_STORE", "workflowstatestore"),
            key_prefix="llm.orchestrator:",
        ),
    )
    registry = AgentRegistryConfig(
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
        ),
        team_name=team_name,
    )
    execution = AgentExecutionConfig(
        max_iterations=int(os.getenv("MAX_ITERATIONS", "1"))
    )

    def on_summary(summary: str):
        print("Journey complete! Summary:", summary, flush=True)

    orchestrator = LLMOrchestrator(
        name=orchestrator_name,
        llm=OpenAIChatClient(),
        pubsub=pubsub,
        state=state,
        registry=registry,
        execution=execution,
        agent_metadata={
            "type": "LLMOrchestrator",
            "description": "LLM-driven Orchestrator",
        },
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "45")),
        runtime=wf.WorkflowRuntime(),
        final_summary_callback=on_summary,
    )

    runner = AgentRunner()
    try:
        runner.serve(orchestrator, port=8004)
    finally:
        runner.shutdown(orchestrator)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
