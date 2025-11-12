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
from dapr_agents.agents.orchestrators.random import RandomOrchestrator
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fellowship.orchestrator.random.app")


def main() -> None:
    orchestrator = RandomOrchestrator(
        name=os.getenv("ORCHESTRATOR_NAME", "FellowshipRandom"),
        pubsub=AgentPubSubConfig(
            pubsub_name=os.getenv("PUBSUB_NAME", "messagepubsub"),
            agent_topic=os.getenv(
                "ORCHESTRATOR_TOPIC", "fellowship.orchestrator.random.requests"
            ),
            broadcast_topic=os.getenv("BROADCAST_TOPIC", "fellowship.broadcast"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(
                store_name=os.getenv("WORKFLOW_STATE_STORE", "workflowstatestore"),
                key_prefix="fellowship.random:",
            ),
        ),
        registry=AgentRegistryConfig(
            store=StateStoreService(
                store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
            ),
            team_name=os.getenv("TEAM_NAME", "fellowship"),
        ),
        execution=AgentExecutionConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "8"))
        ),
        agent_metadata={"legend": "One orchestrator to guide them all."},
        timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "45")),
        runtime=wf.WorkflowRuntime(),
    )
    orchestrator.start()

    runner = AgentRunner()
    try:
        runner.serve(orchestrator, port=8004)
    finally:
        runner.shutdown()
        orchestrator.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
