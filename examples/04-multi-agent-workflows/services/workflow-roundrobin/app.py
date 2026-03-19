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
logger = logging.getLogger("fellowship.orchestrator.roundrobin.app")


def main() -> None:
    orchestrator = DurableAgent(
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
        execution=AgentExecutionConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "4")),
            orchestration_mode=OrchestrationMode.ROUNDROBIN,
        ),
        agent_metadata={"legend": "Sends tasks in a fair rotation."},
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
