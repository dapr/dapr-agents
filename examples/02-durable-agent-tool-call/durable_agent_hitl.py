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
Human-in-the-loop (HITL) example for DurableAgent.

This script shows how to:
  1. Register a before_tool_call hook that returns RequireApproval for specific tools.
  2. Run a DurableAgent with that hook — no decorator changes needed on any tool.
  3. Poll the agent's in-process pending-approval list to detect when the workflow pauses.
  4. Send the human decision back to resume the waiting workflow.

The hook approach works for all tool sources — anything loaded at runtime.

The script will:
  1. Start the agent workflow.
  2. Poll agent.list_pending_approvals() until an approval request appears.
  3. Print the tool name and arguments.
  4. Auto-approve after 5 s (set APPROVED = False in main() to test denial).
  5. Resume or skip the tool based on the decision.
  6. Print the final agent response.
"""

import asyncio
import logging
import time

from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.hooks import Hooks, HookContext, HookDecision, RequireApproval, Proceed
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    import random

    temperature = random.randint(60, 85)
    return f"{location}: {temperature}F, partly cloudy."


@tool
def delete_old_data(dataset: str) -> str:
    """Permanently delete a dataset by name."""
    return f"Dataset '{dataset}' has been deleted."


# the hook decides at runtime which tools need approval
def before_tool(ctx: HookContext) -> HookDecision:
    if ctx.step_name == "DeleteOldData":
        return RequireApproval(
            timeout_seconds=120,
            instructions=f"confirm deletion of dataset: {ctx.payload.get('dataset')}",
        )
    return Proceed()


async def main():
    logging.basicConfig(level=logging.INFO)

    agent = DurableAgent(
        name="hitl-demo",
        role="Operations Assistant",
        goal="Help the user manage data and fetch information.",
        instructions=[
            "Use delete_old_data when asked to clean up or remove data.",
            "Use get_weather for weather queries.",
        ],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[get_weather, delete_old_data],
        hooks=Hooks(before_tool_call=before_tool),
    )

    runner = AgentRunner()

    prompt = "Please delete the old 'sales-2023' dataset and then tell me the weather in Chicago."

    print("\n--- starting workflow ---\n")

    instance_id = await runner.run(
        agent,
        payload={"task": prompt},
        wait=False,
    )
    print(f"workflow started: instance_id = {instance_id}\n")

    AUTO_APPROVE_DELAY = 5  # seconds to pause before sending the decision
    APPROVED = True  # set to False to test denial path

    # When the workflow pauses at a HITL checkpoint, the agent stores the pending approval
    # in-process. Polling list_pending_approvals() here avoids a separate pub/sub subscriber.
    first = True
    while True:
        # Allow more time on the first check (workflow is warming up); shorter for subsequent ones.
        poll_timeout = 60 if first else 10
        approval_data = None
        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            pending = agent.list_pending_approvals()
            if pending:
                approval_data = pending[0]
                break
            await asyncio.sleep(1)

        if approval_data is None:
            break  # No more pending approvals within the window; workflow has likely finished.

        first = False
        approval_request_id = approval_data["approval_request_id"]
        decision_word = "approved" if APPROVED else "denied"

        print("\n" + "=" * 60)
        print("  APPROVAL REQUIRED")
        print("=" * 60)
        print(f"  tool      : {approval_data['step_name']}")
        print(f"  arguments : {approval_data['tool_arguments']}")
        print(f"  instance  : {approval_data['instance_id']}")
        print(f"  request   : {approval_request_id}")
        if approval_data.get("instructions"):
            print(f"  note      : {approval_data['instructions']}")
        print("=" * 60)
        print(
            f"\n  Auto-{'approving' if APPROVED else 'denying'} in {AUTO_APPROVE_DELAY} s "
            f"(edit APPROVED in durable_agent_hitl.py to change).\n"
        )

        await asyncio.sleep(AUTO_APPROVE_DELAY)

        print(f"  decision: {decision_word}\n")
        agent.raise_approval_event(
            instance_id=approval_data["instance_id"],
            approval_request_id=approval_request_id,
            approved=APPROVED,
            reason=f"demo auto-decision: {decision_word}",
        )

    print("\n--- waiting for workflow to finish ---\n")
    result = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=150,
    )
    if result:
        print(f"\nfinal output:\n{getattr(result, 'serialized_output', result)}\n")
    else:
        print(
            "\nworkflow did not complete within the wait window.\n"
            "If the approval event was sent, the workflow is still running.\n"
            f"Re-send manually with:\n"
            f"  python approval_sender.py {instance_id} <approval_request_id> approve\n"
        )

    runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
