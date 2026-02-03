from __future__ import annotations

import logging
import time

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
runtime = wf.WorkflowRuntime()


@runtime.workflow(name="chained_planner_workflow")
def chained_planner_workflow(ctx: DaprWorkflowContext, user_msg: str) -> str:
    """Plan a 3-day trip using chained agent activities."""
    dest = yield ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": user_msg},
        app_id="extractor",
    )
    outline = yield ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": dest.get("content")},
        app_id="planner",
    )
    itinerary = yield ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": outline.get("content")},
        app_id="expander",
    )
    return itinerary.get("content")


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
    state = client.wait_for_workflow_completion(instance_id, timeout_in_seconds=60)

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
