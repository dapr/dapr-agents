#!/usr/bin/env python3
from __future__ import annotations

import logging
import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from phoenix.otel import register

from dapr_agents import Agent
from dapr_agents.llm.huggingface import HFHubChatClient
from dapr_agents.llm.nvidia import NVIDIAChatClient
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.observability import DaprAgentsInstrumentor
from dapr_agents.workflow.decorators import agent_activity

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tracer_provider = register(
    project_name="dapr-weather-agents",
    protocol="http/protobuf",
)
instrumentor = DaprAgentsInstrumentor()
instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

runtime = wf.WorkflowRuntime()

openai_llm = OpenAIChatClient(model="gpt-4o-mini")
nvidia_llm = NVIDIAChatClient(model="meta/llama-3.1-8b-instruct")
hf_llm = HFHubChatClient(model="HuggingFaceTB/SmolLM3-3B")


extractor = Agent(
    name="DestinationExtractor",
    role="Extract destination",
    instructions=["Extract the main city from the user query"],
    llm=openai_llm,
)

planner = Agent(
    name="PlannerAgent",
    role="Outline planner",
    instructions=["Generate a 3-day outline for the destination"],
    llm=nvidia_llm,
)

expander = Agent(
    name="ItineraryAgent",
    role="Itinerary expander",
    instructions=["Expand the outline into a detailed plan"],
    llm=hf_llm,
)


@runtime.workflow(name="chained_planner_workflow")
def chained_planner_workflow(ctx: DaprWorkflowContext, user_msg: str) -> str:
    dest = yield ctx.call_activity(extract_destination, input=user_msg)
    outline = yield ctx.call_activity(plan_outline, input=dest["content"])
    itinerary = yield ctx.call_activity(expand_itinerary, input=outline["content"])
    return itinerary["content"]


@runtime.activity(name="extract_destination")
@agent_activity(agent=extractor)
def extract_destination(ctx) -> dict:
    pass


@runtime.activity(name="plan_outline")
@agent_activity(agent=planner)
def plan_outline(ctx) -> dict:
    pass


@runtime.activity(name="expand_itinerary")
@agent_activity(agent=expander)
def expand_itinerary(ctx) -> dict:
    pass


if __name__ == "__main__":
    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    user_input = "Plan a trip to Paris."

    logger.info("Starting workflow: %s", user_input)
    instance_id = client.schedule_new_workflow(
        workflow=chained_planner_workflow,
        input=user_input,
    )

    logger.info("Workflow started: %s", instance_id)
    state = client.wait_for_workflow_completion(instance_id)

    if not state:
        logger.error("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        logger.info("Trip Itinerary:\n%s", state.serialized_output)
    else:
        logger.error("Workflow ended with status: %s", state.runtime_status)
        if state.failure_details:
            fd = state.failure_details
            logger.error("Failure type: %s", fd.error_type)
            logger.error("Failure message: %s", fd.message)
            logger.error("Stack trace:\n%s", fd.stack_trace)
        else:
            logger.error("Custom status: %s", state.serialized_custom_status)

    runtime.shutdown()
