"""Pod 3: OTEL configured via Dapr state store (agent-runtime) — no OTEL code here."""

import logging
import sys

sys.path.insert(0, "/app/agents")
from common_tools import tools

from dapr_agents import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.runners import AgentRunner

logging.basicConfig(level=logging.INFO)


def main() -> None:
    agent = DurableAgent(
        name="otel-statestore",
        role="Mission Status Reporter",
        goal="Report the current mission status using available tools.",
        instructions=[
            "Use the get_mission_status tool to answer mission status queries."
        ],
        tools=tools,
        llm=DaprChatClient(component_name="llm-provider"),
    )

    runner = AgentRunner()
    try:
        runner.serve(agent, port=8080)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
