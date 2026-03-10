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

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import AgentMemoryConfig
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.workflow.runners.agent import AgentRunner
from dotenv import load_dotenv

load_dotenv()
llm = DaprChatClient(component_name="llm-provider")


def main():
    expert_agent = DurableAgent(
        name="expert_agent",
        role="Technical Support Specialist",
        goal="Provide recommendations based on customer context and issue.",
        instructions=[
            "Provide a clear, actionable recommendation to resolve the issue.",
        ],
        llm=llm,
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="agent-memory",
            )
        ),
    )
    runner = AgentRunner()
    try:
        runner.serve(expert_agent, port=8002)
    finally:
        runner.shutdown(expert_agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down expert agent...")
        exit(0)
