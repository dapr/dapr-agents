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
Drives the Claude-backed DurableAgent through three workflow runs that share
one Claude session:

1. A tool call: Claude calls the dapr-agents ``get_weather`` tool.
2. Session resume: a new workflow run with the same ``session_id`` resumes
   the Claude session from the Dapr state store and remembers turn 1.
3. Human approval: Claude asks to call ``transfer_money``; the run pauses
   until an approval decision is raised as a workflow event.

Environment:
    SESSION_ID     Claude session to use. Defaults to a fresh UUID; pass the
                   id printed by an earlier run to resume that session.
    APPROVAL_MODE  "approve" (default) or "deny" sends the decision from this
                   script; "manual" waits for approval_sender.py or curl.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from dapr.ext.workflow import DaprWorkflowClient
from dotenv import load_dotenv

from agent import build_agent
from dapr_agents import DurableAgent
from dapr_agents.agents.schemas import ApprovalResponseEvent
from dapr_agents.workflow.runners import AgentRunner

RUN_TIMEOUT_SECONDS = 180
APPROVAL_WAIT_SECONDS = 120
APPROVAL_MODES = ("approve", "deny", "manual")


def print_result(title: str, result: Any) -> None:
    print(f"\n=== {title} ===", flush=True)
    print(result, flush=True)
    print("=" * (len(title) + 8) + "\n", flush=True)


async def wait_for_approval_request(
    agent: DurableAgent, instance_id: str
) -> Optional[Dict[str, Any]]:
    """Poll the agent until the workflow publishes its approval request."""
    deadline = time.monotonic() + APPROVAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        for pending in agent.list_pending_approvals():
            if pending.get("instance_id") == instance_id:
                return pending
        await asyncio.sleep(1)
    return None


def send_decision(instance_id: str, approval_request_id: str, approved: bool) -> None:
    """Raise the approval event the paused workflow is waiting for."""
    response = ApprovalResponseEvent(
        approval_request_id=approval_request_id,
        approved=approved,
        reason="approved in app.py" if approved else "rejected in app.py",
    )
    DaprWorkflowClient().raise_workflow_event(
        instance_id=instance_id,
        event_name=f"approval_response_{approval_request_id}",
        data=response.model_dump(mode="json"),
    )


async def run_approval_turn(
    runner: AgentRunner, agent: DurableAgent, session_id: str, mode: str
) -> None:
    instance_id = await runner.run(
        agent,
        payload={"task": "Send $25 to Alice for dinner.", "session_id": session_id},
        wait=False,
    )
    request = await wait_for_approval_request(agent, instance_id)
    if request is None:
        print("No approval request appeared; did Claude call transfer_money?")
        return

    request_id = request["approval_request_id"]
    print("\n=== Approval required ===", flush=True)
    print(f"  tool        : {request['step_name']}", flush=True)
    print(f"  arguments   : {request['tool_arguments']}", flush=True)
    print(f"  instance_id : {instance_id}", flush=True)
    print(f"  request_id  : {request_id}", flush=True)
    if mode == "manual":
        print(
            f"\n  Send a decision:\n"
            f"    python approval_sender.py {instance_id} {request_id} approve\n",
            flush=True,
        )
    else:
        send_decision(instance_id, request_id, approved=mode == "approve")
        print(f"  decision    : {mode}\n", flush=True)

    state = await asyncio.to_thread(
        runner.wait_for_workflow_completion,
        instance_id,
        timeout_in_seconds=APPROVAL_WAIT_SECONDS + RUN_TIMEOUT_SECONDS,
    )
    print_result("Turn 3: after approval", getattr(state, "serialized_output", state))


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    session_id = os.getenv("SESSION_ID") or str(uuid.uuid4())
    mode = os.getenv("APPROVAL_MODE", "approve").lower()
    if mode not in APPROVAL_MODES:
        raise SystemExit(f"APPROVAL_MODE must be one of {', '.join(APPROVAL_MODES)}")
    print(f"\nClaude session_id: {session_id}\n", flush=True)

    agent = build_agent()
    runner = AgentRunner()
    try:
        result = await runner.run(
            agent,
            payload={"task": "What's the weather in Paris?", "session_id": session_id},
            timeout_in_seconds=RUN_TIMEOUT_SECONDS,
        )
        print_result("Turn 1: tool call", result)

        result = await runner.run(
            agent,
            payload={
                "task": "Which city did I just ask about? Answer in one sentence.",
                "session_id": session_id,
            },
            timeout_in_seconds=RUN_TIMEOUT_SECONDS,
        )
        print_result("Turn 2: resumed session", result)

        await run_approval_turn(runner, agent, session_id, mode)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
