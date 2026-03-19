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

import dapr.ext.workflow as wf
from dapr_agents import call_agent

wfr = wf.WorkflowRuntime()
WEATHER_AGENT_APP_ID = "weather-agent"


@wfr.workflow
def orchestration_workflow(ctx: wf.DaprWorkflowContext):
    """Calls WeatherAgent as a child workflow step."""
    result = yield call_agent(
        ctx,
        "WeatherAgent",
        input={"task": "What is the weather in London?"},
        app_id=WEATHER_AGENT_APP_ID,
    )
    return result


def main() -> None:
    wfr.start()
    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(workflow=orchestration_workflow)
    print(f"Scheduled orchestration workflow instance: {instance_id}")

    state = client.wait_for_workflow_completion(
        instance_id=instance_id,
        timeout_in_seconds=120,
        fetch_payloads=True,
    )
    print(f"Workflow status: {state.runtime_status.name}")
    if state.serialized_output:
        print(f"Result: {state.serialized_output}")

    wfr.shutdown()


if __name__ == "__main__":
    main()
