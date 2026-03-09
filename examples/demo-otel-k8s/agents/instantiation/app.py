"""Pod 1: OTEL configured via AgentObservabilityConfig in code."""

import logging
import sys

sys.path.insert(0, "/app/agents")
from common_tools import tools

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import AgentObservabilityConfig, AgentTracingExporter
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.runners import AgentRunner

logging.basicConfig(level=logging.INFO)


def main() -> None:
    agent = DurableAgent(
        name="otel-instantiation",
        role="Mission Status Reporter",
        goal="Report the current mission status using available tools.",
        instructions=[
            "Use the get_mission_status tool to answer mission status queries."
        ],
        tools=tools,
        llm=DaprChatClient(component_name="llm-provider"),
        agent_observability=AgentObservabilityConfig(
            enabled=True,
            tracing_enabled=True,
            tracing_exporter=AgentTracingExporter.OTLP_GRPC,
            endpoint="http://dapr-agents-opentelemetry-collector.dapr-agents.svc.cluster.local:4317",
            service_name="otel-instantiation-agent",
        ),
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
