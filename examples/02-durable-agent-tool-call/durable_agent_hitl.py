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
  1. Define a tool that requires human approval before it runs.
  2. Run a DurableAgent with approval enabled.
  3. Receive the ApprovalRequiredEvent via Dapr Pub/Sub.
  4. Send the human decision back to resume the waiting workflow.

Start the Dapr sidecar alongside this script
The script will:
  1. Start the agent workflow.
  2. Receive the ApprovalRequiredEvent from the pub/sub topic.
  3. Print the tool name and arguments.
  4. Auto-approve after 5 s (set APPROVED = False in main() to test denial).
  5. Resume or skip the tool based on the decision.
  6. Print the final agent response.
"""

import asyncio
import json
import logging

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request, Response

from dapr_agents import DurableAgent, tool, AgentApprovalConfig
from dapr_agents.agents.configs import AgentExecutionConfig
from dapr_agents.agents.schemas import ApprovalRequiredEvent
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)

# both Uvicorn and main() run in the same event loop (via asyncio.create_task), so put_nowait() from the FastAPI handler immediately wakes up get() in main()
_approval_queue: asyncio.Queue = asyncio.Queue()
_dapr_app = FastAPI()


@_dapr_app.get("/dapr/subscribe")
def _subscribe():
    return [
        {
            "pubsubname": "messagepubsub",
            "topic": "agent-approval-requests",
            "route": "/agent-approval-requests",
        }
    ]


@_dapr_app.post("/agent-approval-requests")
async def _receive_approval_request(request: Request):
    """Called by Dapr when the workflow publishes an approval request.
    Parses the CloudEvent and pushes the typed payload to the queue for main() to handle.
    """
    body = await request.body()
    envelope = json.loads(body)
    data = envelope.get("data") or envelope
    event = ApprovalRequiredEvent(**data)
    _approval_queue.put_nowait(event)
    logger.info(
        "Approval request received via pub/sub: tool='%s' request_id=%s instance=%s",
        event.tool_name,
        event.approval_request_id,
        event.instance_id,
    )
    return Response(status_code=200)


# mock tools
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    import random

    temperature = random.randint(60, 85)
    return f"{location}: {temperature}F, partly cloudy."


@tool(requires_approval=True, approval_timeout_seconds=120)
def delete_old_data(dataset: str) -> str:
    """Permanently delete a dataset by name."""
    return f"Dataset '{dataset}' has been deleted."


# main
async def main():
    logging.basicConfig(level=logging.INFO)

    # run Uvicorn as a task in this event loop so the pub/sub handler and queue.get() share the same loop.
    _uvicorn_config = uvicorn.Config(
        _dapr_app, host="0.0.0.0", port=8001, log_level="warning"
    )
    _uvicorn_server = uvicorn.Server(_uvicorn_config)
    _uvicorn_task = asyncio.create_task(_uvicorn_server.serve())
    await asyncio.sleep(
        0.5
    )  # wait for Uvicorn to open the socket before Dapr delivers messages

    approval_config = AgentApprovalConfig(
        enabled=True,
        pubsub_name="messagepubsub",
        topic="agent-approval-requests",
        default_timeout_seconds=120,
    )

    agent = DurableAgent(
        name="hitl-demo",
        role="Operations Assistant",
        goal="Help the user manage data and fetch information.",
        instructions=[
            "Use delete_old_data when asked to clean up or remove data.",
            "Use get_weather for weather queries.",
        ],
        llm=OpenAIChatClient(model="openai/gpt-4o-mini"),
        tools=[get_weather, delete_old_data],
        execution=AgentExecutionConfig(approval=approval_config),
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

    # block until the workflow publishes an ApprovalRequiredEvent and Dapr delivers it here
    print("waiting for approval request from pub/sub (workflow is paused) ...\n")
    try:
        approval_event: ApprovalRequiredEvent = await asyncio.wait_for(
            _approval_queue.get(), timeout=60.0
        )
    except asyncio.TimeoutError:
        print(
            "\n[ERROR] No approval request received within 60 s.\n"
            "Check that the Dapr sidecar is running (--app-port 8001) and that\n"
            "Redis pub/sub is reachable.\n"
        )
        return

    AUTO_APPROVE_DELAY = 5  # seconds to pause before sending the decision
    APPROVED = True  # set to False to test denial path

    print("\n" + "=" * 60)
    print("  APPROVAL REQUIRED")
    print("=" * 60)
    print(f"  tool      : {approval_event.tool_name}")
    print(f"  arguments : {approval_event.tool_arguments}")
    print(f"  instance  : {approval_event.instance_id}")
    print(f"  request   : {approval_event.approval_request_id}")
    print("=" * 60)
    decision_word = "approved" if APPROVED else "denied"
    print(
        f"\n  Auto-{'approving' if APPROVED else 'denying'} in {AUTO_APPROVE_DELAY} s "
        f"(edit APPROVED in durable_agent_hitl.py to change).\n"
    )

    await asyncio.sleep(AUTO_APPROVE_DELAY)

    print(f"  decision: {decision_word}\n")
    agent.raise_approval_event(
        instance_id=approval_event.instance_id,
        approval_request_id=approval_event.approval_request_id,
        approved=APPROVED,
        reason=f"demo auto-decision: {decision_word}",
    )

    while True:
        try:
            next_event: ApprovalRequiredEvent = await asyncio.wait_for(
                _approval_queue.get(), timeout=10.0
            )
        except asyncio.TimeoutError:
            break

        print("\n" + "=" * 60)
        print("  ANOTHER APPROVAL REQUIRED")
        print("=" * 60)
        print(f"  tool      : {next_event.tool_name}")
        print(f"  arguments : {next_event.tool_arguments}")
        print("=" * 60)
        print(
            f"\n  Auto-{'approving' if APPROVED else 'denying'} in {AUTO_APPROVE_DELAY} s ...\n"
        )
        await asyncio.sleep(AUTO_APPROVE_DELAY)
        print(f"  decision: {decision_word}\n")
        agent.raise_approval_event(
            instance_id=next_event.instance_id,
            approval_request_id=next_event.approval_request_id,
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
            f"  python approval_sender.py {instance_id} "
            f"{approval_event.approval_request_id} approve\n"
        )

    _uvicorn_server.should_exit = True
    try:
        await _uvicorn_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
