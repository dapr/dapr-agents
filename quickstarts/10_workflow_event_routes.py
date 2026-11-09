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

"""Signal a running workflow from a pub/sub message with a WorkflowEventRouteSpec.

The script starts a workflow that waits for a ``job_finished`` external event,
publishes a ``JobFinished`` message to the ``jobs.finished`` topic, and the
event route raises the event on the workflow named in the message. No LLM is
needed. It exits 0 when the workflow completed with the event payload.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import timedelta

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from pydantic import BaseModel

from dapr_agents.workflow import (
    MessageContext,
    WorkflowEventRouteSpec,
    WorkflowEventTarget,
)
from dapr_agents.workflow.utils.registration import register_message_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PUBSUB_NAME = "agent-pubsub"
TOPIC = "jobs.finished"
EVENT_NAME = "job_finished"
WAIT_SECONDS = 60


class JobRef(BaseModel):
    workflow_id: str


class JobFinished(BaseModel):
    job: JobRef
    status: str
    result: str


runtime = wf.WorkflowRuntime()


@runtime.workflow(name="job_workflow")
def job_workflow(ctx: wf.DaprWorkflowContext, job_name: str):
    """Wait for the job's completion signal, or give up after a timeout."""
    event = ctx.wait_for_external_event(EVENT_NAME)
    timeout = ctx.create_timer(timedelta(seconds=WAIT_SECONDS))
    winner = yield wf.when_any([event, timeout])
    if winner == timeout:
        return {"job": job_name, "status": "timed_out"}
    payload = event.get_result()
    return {"job": job_name, "status": payload["status"], "result": payload["result"]}


def authorize(
    message: JobFinished, ctx: MessageContext, target: WorkflowEventTarget
) -> bool:
    """Only signal workflows this app owns, whatever the publisher claims."""
    return target.instance_id.startswith("job-")


route = WorkflowEventRouteSpec(
    pubsub_name=PUBSUB_NAME,
    topic=TOPIC,
    event_name=EVENT_NAME,
    instance_id_from="job.workflow_id",
    message_model=JobFinished,
    authorize=authorize,
)


def main() -> int:
    runtime.start()
    wf_client = wf.DaprWorkflowClient()
    try:
        with DaprClient() as client:
            closers = register_message_routes(
                dapr_client=client, routes=[route], wf_client=wf_client
            )
            try:
                instance_id = f"job-{uuid.uuid4().hex[:8]}"
                wf_client.schedule_new_workflow(
                    workflow=job_workflow,
                    input="nightly-report",
                    instance_id=instance_id,
                )
                wf_client.wait_for_workflow_start(instance_id, timeout_in_seconds=30)
                logger.info("Workflow %s is waiting for %r", instance_id, EVENT_NAME)

                message = JobFinished(
                    job=JobRef(workflow_id=instance_id),
                    status="done",
                    result="42 rows exported",
                )
                # Give the streaming subscription a moment to attach.
                time.sleep(2)
                client.publish_event(
                    pubsub_name=PUBSUB_NAME,
                    topic_name=TOPIC,
                    data=message.model_dump_json(),
                    data_content_type="application/json",
                )
                logger.info("Published JobFinished for %s to %s", instance_id, TOPIC)

                state = wf_client.wait_for_workflow_completion(
                    instance_id, timeout_in_seconds=WAIT_SECONDS + 30
                )
            finally:
                for close in closers:
                    close()
    finally:
        runtime.shutdown()

    if state is None:
        print("Workflow did not complete in time.")
        return 1
    output = json.loads(state.serialized_output or "{}")
    print(f"Workflow {instance_id} finished with {state.runtime_status.name}: {output}")
    if output.get("status") != "done":
        return 1
    print("Event route delivered the event.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
