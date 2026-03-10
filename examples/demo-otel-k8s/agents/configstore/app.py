"""Pod 4: OTEL configured via Dapr Configuration Store — hot-reloadable OTel settings."""

import logging
import sys

sys.path.insert(0, "/app/agents")
from common_tools import tools

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    RuntimeConfigKey,
    RuntimeSubscriptionConfig,
)
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.runners import AgentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def on_config_change(key: str, value: object) -> None:
    """Callback invoked after each successful config update."""
    safe: object = "***" if "header" in key.lower() else value
    logger.info("Config changed: %s = %s", key, safe)


def main() -> None:
    agent = DurableAgent(
        name="otel-configstore",
        role="Mission Status Reporter",
        goal="Report the current mission status using available tools.",
        instructions=[
            "Use the get_mission_status tool to answer mission status queries."
        ],
        tools=tools,
        llm=DaprChatClient(component_name="llm-provider"),
        configuration=RuntimeSubscriptionConfig(
            store_name="otel-config",
            keys=[
                RuntimeConfigKey.OTEL_SDK_DISABLED,
                RuntimeConfigKey.OTEL_EXPORTER_OTLP_ENDPOINT,
                RuntimeConfigKey.OTEL_EXPORTER_OTLP_HEADERS,
                RuntimeConfigKey.OTEL_SERVICE_NAME,
                RuntimeConfigKey.OTEL_TRACING_ENABLED,
                RuntimeConfigKey.OTEL_TRACES_EXPORTER,
                RuntimeConfigKey.OTEL_LOGGING_ENABLED,
                RuntimeConfigKey.OTEL_LOGS_EXPORTER,
            ],
            on_config_change=on_config_change,
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
