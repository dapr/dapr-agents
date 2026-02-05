import logging
import os

from dotenv import load_dotenv

from dapr_agents.agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    OrchestrationMode,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fellowship.orchestrator.random.app")


def main() -> None:
    orchestrator = DurableAgent(
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
            max_iterations=int(os.getenv("MAX_ITERATIONS", "1")),
            orchestration_mode=OrchestrationMode.RANDOM,
        ),
        agent_metadata={"legend": "One orchestrator to guide them all."},
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
