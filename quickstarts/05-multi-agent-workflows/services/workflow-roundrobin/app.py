import logging
import os

from dotenv import load_dotenv

import dapr.ext.workflow as wf
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.orchestrators.roundrobin import RoundRobinOrchestrator
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("fellowship.orchestrator.roundrobin.app")


def main() -> None:
    orchestrator = RoundRobinOrchestrator(
        name=os.getenv("ORCHESTRATOR_NAME", "FellowshipRoundRobin"),
        pubsub=AgentPubSubConfig(
            pubsub_name=os.getenv("PUBSUB_NAME", "messagepubsub"),
            agent_topic=os.getenv(
                "ORCHESTRATOR_TOPIC", "fellowship.orchestrator.roundrobin.requests"
            ),
            broadcast_topic=os.getenv("BROADCAST_TOPIC", "fellowship.broadcast"),
        ),
        state=AgentStateConfig(
            store=StateStoreService(
                store_name=os.getenv("WORKFLOW_STATE_STORE", "workflowstatestore"),
                key_prefix="fellowship.roundrobin:",
            ),
        ),
        registry=AgentRegistryConfig(
            store=StateStoreService(
                store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
            ),
            team_name=os.getenv("TEAM_NAME", "fellowship"),
        ),
        agent_metadata={"legend": "Sends tasks in a fair rotation."},
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
