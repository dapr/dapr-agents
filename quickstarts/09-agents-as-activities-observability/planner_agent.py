from __future__ import annotations

import logging

from dotenv import load_dotenv

from dapr_agents.agents.configs import AgentObservabilityConfig, AgentTracingExporter
from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
llm = DaprChatClient(component_name="openai")


def main():
    planner = DurableAgent(
        name="PlannerAgent",
        role="Trip planner",
        instructions=[
            "Create a concise 3-day outline for the given destination.",
            "Balance culture, food, and leisure activities.",
        ],
        llm=llm,
        agent_observability=AgentObservabilityConfig(
            enabled=True,
            tracing_enabled=True,
            tracing_exporter=AgentTracingExporter.ZIPKIN,
            endpoint="http://localhost:9411/api/v2/spans",
        ),
    )

    runner = AgentRunner()
    try:
        runner.serve(planner, port=8002)
    finally:
        runner.shutdown(planner)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down agent...")
