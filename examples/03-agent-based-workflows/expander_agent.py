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
    expander = DurableAgent(
        name="ItineraryAgent",
        role="Itinerary expander",
        llm=llm,
        instructions=[
            "Expand a 3-day outline into a detailed itinerary.",
            "Include Morning, Afternoon, and Evening sections each day.",
        ],
    )

    runner = AgentRunner()
    try:
        runner.serve(expander, port=8003)
    finally:
        runner.shutdown(expander)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down agent...")
