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

"""
HITL via Pub/Sub: the agent publishes ApprovalRequiredEvent to a Dapr topic.
An external subscriber (a Slack bot, dashboard, or the companion script below)
reads from that topic and sends back the decision.

How it works
------------
1. The before_tool hook returns RequireApproval for the delete_old_data tool.
2. The agent publishes an ApprovalRequiredEvent to the topic configured in
   AgentApprovalConfig (pubsub_name + topic).
3. The workflow suspends and waits for an approval_response_{id} event.
4. An external system subscribes to the topic, presents the request to a human,
   and calls DaprWorkflowClient.raise_workflow_event() to resume the workflow.

To test locally, run this script in one terminal, then run hitl_wf_event.py in
a second terminal passing the instance_id and approval_request_id printed here.

Prerequisites
-------------
- dapr sidecar running (dapr run ...)
- a Dapr pub/sub component named "pubsub" pointing at Redis or similar
"""

import asyncio
import logging

from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentApprovalConfig, AgentExecutionConfig
from dapr_agents.hooks import Hooks, HookContext, HookDecision, RequireApproval, Proceed
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)


@tool
def delete_old_data(dataset: str) -> str:
    """Permanently delete a dataset by name."""
    return f"Dataset '{dataset}' has been deleted."


def before_tool(ctx: HookContext) -> HookDecision:
    if ctx.step_name == "DeleteOldData":
        return RequireApproval(
            timeout_seconds=300,
            instructions=f"Confirm deletion of dataset: {ctx.payload.get('dataset')}",
        )
    return Proceed()


async def main():
    logging.basicConfig(level=logging.INFO)

    agent = DurableAgent(
        name="hitl-pubsub-demo",
        role="Operations Assistant",
        goal="Help the user manage data.",
        instructions=["Use delete_old_data when asked to clean up or remove data."],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[delete_old_data],
        hooks=Hooks(before_tool_call=[before_tool]),
        execution=AgentExecutionConfig(
            approval=AgentApprovalConfig(
                pubsub_name="pubsub",
                topic="agent-approval-requests",
            )
        ),
    )

    runner = AgentRunner()

    instance_id = await runner.run(
        agent,
        payload={"task": "Please delete the old 'sales-2023' dataset."},
        wait=False,
    )

    print(f"\nWorkflow started: instance_id = {instance_id}")
    print(
        "\nThe agent has published an ApprovalRequiredEvent to the 'agent-approval-requests' topic."
    )
    print("Your subscriber should receive it and prompt a human for a decision.")
    print("\nTo approve or deny manually, run in a separate terminal:")
    print(f"  python hitl_wf_event.py {instance_id} <approval_request_id> approve\n")

    result = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=360,
    )
    if result:
        print(f"\nFinal output:\n{getattr(result, 'serialized_output', result)}\n")
    else:
        print("\nWorkflow did not complete within the wait window.")

    runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
