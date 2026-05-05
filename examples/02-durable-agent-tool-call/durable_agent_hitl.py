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
Example 1 — HITL via HTTP.

Agent side : the workflow pauses and holds the approval request in memory,
             exposed via GET /hitl/approvals (mounted automatically by AgentRunner).
Client side: a human polls GET /hitl/approvals and sends the decision back
             with POST /hitl/approvals/{approval_request_id}/respond.

No pub/sub component or Dapr sidecar required for the approval round-trip —
just the agent's HTTP server.

This script runs both sides in the same process for simplicity.
In production the client call would come from a separate service or dashboard.

Other patterns:
  - hitl_pubsub.py    — approval round-trip over Dapr pub/sub
  - hitl_wf_event.py  — approval round-trip via direct workflow event
"""

import asyncio
import logging
import time

import httpx
from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.hooks import Hooks, HookContext, HookDecision, RequireApproval, Proceed
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)

# Port the AgentRunner HTTP server will listen on.
AGENT_PORT = 8080
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
        name="hitl-http-demo",
        role="Operations Assistant",
        goal="Help the user manage data.",
        instructions=["Use delete_old_data when asked to clean up or remove data."],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[delete_old_data],
        hooks=Hooks(before_tool_call=[before_tool]),
    )

    runner = AgentRunner()

    # serve() starts the FastAPI HTTP server which mounts GET/POST /hitl/approvals.
    # We run it as a background task so this script can keep driving the client side.
    server_task = asyncio.create_task(
        runner.serve(agent, port=AGENT_PORT, host="127.0.0.1")
    )

    # Give the server a moment to come up.
    await asyncio.sleep(2)

    instance_id = await runner.run(
        agent,
        payload={"task": "Please delete the old 'sales-2023' dataset."},
        wait=False,
    )
    print(f"\nWorkflow started: instance_id = {instance_id}\n")

    # --- client side: poll GET /hitl/approvals until the request appears ---
    base_url = f"http://127.0.0.1:{AGENT_PORT}"
    approval_data = None
    deadline = time.monotonic() + 60
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            resp = await client.get(f"{base_url}/hitl/approvals")
            pending = resp.json()
            if pending:
                approval_data = pending[0]
                break
            await asyncio.sleep(1)

        if approval_data is None:
            print("No approval request appeared within the wait window.")
            runner.shutdown(agent)
            server_task.cancel()
            return

        approval_request_id = approval_data["approval_request_id"]
        print(f"  tool      : {approval_data['step_name']}")
        print(f"  arguments : {approval_data['tool_arguments']}")
        print(f"  request   : {approval_request_id}")
        print(f"\n  Sending HTTP decision: {'approve' if APPROVED else 'deny'}\n")

        # --- client side: POST the decision back via HTTP ---
        await client.post(
            f"{base_url}/hitl/approvals/{approval_request_id}/respond",
            json={"approved": APPROVED, "reason": "approved via HTTP example"},
        )

    result = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=150,
    )
    if result:
        print(f"\nFinal output:\n{getattr(result, 'serialized_output', result)}\n")
    else:
        print("\nWorkflow did not complete within the wait window.")

    runner.shutdown(agent)
    server_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
