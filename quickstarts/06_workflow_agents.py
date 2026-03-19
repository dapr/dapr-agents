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

import time

import dapr.ext.workflow as wf

from dapr_agents.workflow.utils.core import call_agent

wfr = wf.WorkflowRuntime()


@wfr.workflow(name="support_workflow")
def support_workflow(ctx: wf.DaprWorkflowContext, request: dict) -> str:
    """Process a support request through triage and expert agents."""
    # Each step is durable and can be retried.
    triage_result = yield call_agent(
        ctx,
        "triage_agent",
        input={"task": f"Assist with the following support request:\n\n{request}"},
        app_id="triage-agent",
    )
    if triage_result:
        print("Triage result:", triage_result.get("content", ""), flush=True)

    recommendation = yield call_agent(
        ctx,
        "expert_agent",
        input={"task": triage_result.get("content", "")},
        app_id="expert-agent",
    )
    if recommendation:
        print("Recommendation:", recommendation.get("content", ""), flush=True)

    return recommendation.get("content", "") if recommendation else ""


if __name__ == "__main__":
    wfr.start()
    time.sleep(5)  # give the runtime time to initialize

    client = wf.DaprWorkflowClient()
    request = {
        "customer": "alice",
        "issue": "Unable to access dashboard after recent update",
    }
    instance_id = client.schedule_new_workflow(
        workflow=support_workflow,
        input=request,
    )
    print(f"Workflow started: {instance_id}", flush=True)

    state = client.wait_for_workflow_completion(instance_id, timeout_in_seconds=60)
    if not state:
        print("No workflow state returned.")
    elif state.runtime_status.name == "COMPLETED":
        print(f"\n✅ Final Recommendation:\n{state.serialized_output}")
    else:
        print(f"Workflow finished with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)

    wfr.shutdown()
