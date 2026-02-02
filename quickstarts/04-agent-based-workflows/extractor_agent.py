from __future__ import annotations

import logging

from dotenv import load_dotenv

from dapr_agents.workflow.runners.agent import AgentRunner
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
llm = DaprChatClient(component_name="openai")


def main():
    extractor = DurableAgent(
        name="DestinationExtractor",
        role="Extract destination",
        instructions=[
            "Extract the main city from the user's message.",
            "Return only the city name, nothing else.",
        ],
        llm=llm,
    )

    runner = AgentRunner()
    try:
        runner.serve(extractor, port=8001)
    finally:
        runner.shutdown(extractor)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down agent...")
