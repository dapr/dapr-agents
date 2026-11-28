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

from pathlib import Path

from dapr_agents import AgentRunner, DurableAgent, FileSystemSkillSource
from dapr_agents.agents.configs import AgentExecutionConfig, AgentStateConfig
from dapr_agents.llm import DaprChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types.agent import ToolExecutionMode


SKILLS_PATH = Path(__file__).resolve().parent / "skills"


def main() -> None:
    shipment_agent = DurableAgent(
        name="ShipmentExceptionAgent",
        role="Shipment Exception Specialist",
        goal="Route delayed shipment exceptions using the local shipment policy.",
        instructions=[
            "For delayed shipment exceptions, use the configured shipment-exceptions "
            "skill before answering.",
            "Recommend a course of action with supporting evidence and any missing facts.",
        ],
        llm=DaprChatClient(component_name="llm-provider"),
        execution=AgentExecutionConfig(
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        ),
        state=AgentStateConfig(
            store=StateStoreService(store_name="agent-workflow"),
        ),
        skills=[FileSystemSkillSource(skill_paths=[SKILLS_PATH])],
    )

    runner = AgentRunner()
    try:
        runner.serve(shipment_agent, port=8001)
    finally:
        runner.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
