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
Example 3 — HITL via direct workflow event.

Agent side : the workflow pauses and waits for an external event named
             approval_response_{approval_request_id}. No pub/sub topic is
             configured — the agent holds the request in memory only.
Client side: a human reads the pending request (printed to stdout here) and
             raises the workflow event directly via DaprWorkflowClient, which
             delivers it straight to the waiting workflow through the Dapr sidecar.

No pub/sub component required. The Dapr sidecar must be reachable.

This script runs both sides in the same process for simplicity.
In production the client call would be a separate CLI command or script.

Other patterns:
  - durable_agent_hitl.py  — approval round-trip over HTTP
  - hitl_pubsub.py         — approval round-trip over Dapr pub/sub
"""

import asyncio
import logging
import time

from dapr.ext.workflow import DaprWorkflowClient
from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.schemas import ApprovalResponseEvent
from dapr_agents.hooks import Hooks, HookContext, HookDecision, RequireApproval, Proceed
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)

APPROVED = True  # set to False to test the denial path


@tool
def delete_old_data(dataset: str) -> str:
    """Permanently delete a dataset by name."""
    return f"Dataset '{dataset}' has been deleted."


def before_tool(ctx: HookContext) -> HookDecision:
    if ctx.step_name == "DeleteOldData":
        return RequireApproval(
            timeout_seconds=120,
            instructions=f"Confirm deletion of dataset: {ctx.payload.get('dataset')}",
        )
    return Proceed()


async def main():
    logging.basicConfig(level=logging.INFO)

    agent = DurableAgent(
        name="hitl-wf-event-demo",
        role="Operations Assistant",
        goal="Help the user manage data.",
        instructions=["Use delete_old_data when asked to clean up or remove data."],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[delete_old_data],
        hooks=Hooks(before_tool_call=[before_tool]),
        # No AgentApprovalConfig pubsub_name — request is held in memory only.
    )

    runner = AgentRunner()

    instance_id = await runner.run(
        agent,
        payload={"task": "Please delete the old 'sales-2023' dataset."},
        wait=False,
    )
    print(f"\nWorkflow started: instance_id = {instance_id}\n")

    # --- client side: poll in-memory list until the request appears ---
    approval_data = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        pending = agent.list_pending_approvals()
        if pending:
            approval_data = pending[0]
            break
        await asyncio.sleep(1)

    if approval_data is None:
        print("No approval request appeared within the wait window.")
        runner.shutdown(agent)
        return

    approval_request_id = approval_data["approval_request_id"]
    print(f"  tool      : {approval_data['step_name']}")
    print(f"  arguments : {approval_data['tool_arguments']}")
    print(f"  instance  : {approval_data['instance_id']}")
    print(f"  request   : {approval_request_id}")
    print(f"\n  Raising workflow event: {'approve' if APPROVED else 'deny'}\n")

    # --- client side: raise the workflow event directly via DaprWorkflowClient ---
    event_name = f"approval_response_{approval_request_id}"
    response = ApprovalResponseEvent(
        approval_request_id=approval_request_id,
        approved=APPROVED,
        reason="approved via workflow event example",
    )
    wf_client = DaprWorkflowClient()
    wf_client.raise_workflow_event(
        instance_id=approval_data["instance_id"],
        event_name=event_name,
        data=response.model_dump(mode="json"),
    )

    result = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=120,
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
