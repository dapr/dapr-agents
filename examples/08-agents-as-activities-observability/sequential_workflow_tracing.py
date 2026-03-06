#!/usr/bin/env python3

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
        workflow="dapr.durableagent.destination-extractor.workflow",
        input={"task": user_msg},
        app_id="extractor",
    )
    outline = yield ctx.call_child_workflow(
        workflow="dapr.durableagent.planner-agent.workflow",
        input={"task": dest.get("content")},
        app_id="planner",
    )
    itinerary = yield ctx.call_child_workflow(
        workflow="dapr.durableagent.itinerary-agent.workflow",
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
