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
Example 4 — HITL via the built-in ``ask_user`` tool.

Unlike the other HITL examples in this folder, this one is not a
``before_tool_call`` approval gate — the LLM itself decides, mid-turn, that it
needs information only the human can provide, and calls the built-in
``ask_user`` tool to get it. This is the pattern a chat app uses to keep
coming back to the *same* running workflow instance across several HTTP
calls, instead of starting a new workflow for every message.

Agent side : enabling ``ask_user`` (via ``AgentExecutionConfig(streaming=True,
             builtin_tools=[BuiltinTool.ASK_USER])``) suspends the workflow on
             ``ctx.wait_for_external_event(...)`` when the tool is called, and
             emits a ``USER_INPUT_REQUESTED`` chunk on the run's stream.
Client side: a chat client starts the run via ``POST /agent/run/stream``,
             watches the stream for that chunk, and answers by calling
             ``POST /agent/run/input`` with the same ``request_id`` /
             ``target_instance_id`` — which resumes that exact instance.

If the client's connection to the stream drops before it answers (e.g. the
chat UI reloads), the question isn't lost: ``GET /agent/instances/{id}``
surfaces it under ``pending_inputs`` so the client can recover the
``request_id`` and still reply via the same ``/input`` endpoint. This script
simulates exactly that: it reads only the first chunk, closes the stream, then
recovers the pending question from the status endpoint before answering.

No pub/sub component required — just the agent's HTTP server.

This script runs both sides in the same process for simplicity. In production
the client call would come from a separate chat front end.

Other patterns:
  - durable_agent_hitl.py  — tool-call approval over HTTP
  - hitl_pubsub.py         — tool-call approval over Dapr pub/sub
  - hitl_wf_event.py       — tool-call approval via direct workflow event
"""

import asyncio
import json
import logging
import threading
import time

import httpx
from dotenv import load_dotenv

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentExecutionConfig, BuiltinTool
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm.openai import OpenAIChatClient

load_dotenv()

logger = logging.getLogger(__name__)

# Port the AgentRunner HTTP server will listen on.
AGENT_PORT = 8082

# Canned reply the "human" gives when the agent asks its clarifying question.
ANSWER = "Room 204"


@tool
def book_meeting_room(room: str, time_slot: str) -> str:
    """Book a meeting room for the given time slot."""
    return f"Booked {room} for {time_slot}."


def _client_flow(base_url: str) -> None:
    """Run the chat-client side of the demo (blocking, plain requests-style flow
    driven by an inner event loop so this script stays a single file)."""

    asyncio.run(_async_client_flow(base_url))


async def _async_client_flow(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # --- start the run and read only the first USER_INPUT_REQUESTED chunk ---
        request_id = None
        target_instance_id = None
        async with client.stream(
            "POST",
            f"{base_url}/agent/run/stream",
            json={"task": "Please book a meeting room for tomorrow at 2pm."},
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk["type"] == "user_input_requested":
                    event = chunk["event_data"]
                    request_id = event["request_id"]
                    target_instance_id = event["target_instance_id"]
                    print(f"\n[agent asks] {event['question']}")
                    break

        if request_id is None:
            print("No question was asked within the stream window.")
            return

        # --- simulate the chat client disconnecting here (page reload, lost ---
        # --- connection, etc.) and coming back later with only the instance id ---
        print("[chat client] (stream closed — simulating a dropped connection)")
        await asyncio.sleep(1)

        status_url = f"{base_url}/agent/instances/{target_instance_id}"
        status = (await client.get(status_url)).json()
        pending = status.get("pending_inputs") or []
        if not pending:
            print("No pending input recovered from the status endpoint.")
            return
        recovered = pending[0]
        print(
            f"[chat client] recovered pending question via GET {status_url}: "
            f"{recovered['question']!r}"
        )

        # --- answer using the recovered request_id / target_instance_id ---
        print(f"[chat client] answering: {ANSWER}\n")
        await client.post(
            f"{base_url}/agent/run/input",
            json={
                "request_id": recovered["request_id"],
                "target_instance_id": recovered["target_instance_id"],
                "answer": ANSWER,
            },
        )

        # --- poll the same instance until the turn completes ---
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = (await client.get(status_url)).json()
            if status.get("runtime_status") == "COMPLETED":
                output = status.get("serialized_output")
                try:
                    output = json.loads(output) if output else output
                except (TypeError, ValueError):
                    pass
                print(f"Final output:\n{output}\n")
                return
            await asyncio.sleep(1)

        print("Workflow did not complete within the wait window.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    agent = DurableAgent(
        name="ask-user-demo",
        role="Scheduling Assistant",
        goal="Help the user book meeting rooms.",
        instructions=[
            "Use book_meeting_room once you know which room to book.",
            "If the user hasn't said which room, use ask_user to ask them "
            "directly instead of guessing.",
        ],
        llm=OpenAIChatClient(model="gpt-4o-mini"),
        tools=[book_meeting_room],
        execution=AgentExecutionConfig(
            streaming=True,
            builtin_tools=[BuiltinTool.ASK_USER],
        ),
    )

    runner = AgentRunner()

    # serve() blocks on uvicorn.run(), so host it on a background thread.
    # In production the agent service and the chat client are separate
    # processes; this script runs both for simplicity.
    server_thread = threading.Thread(
        target=runner.serve,
        kwargs={"agent": agent, "host": "127.0.0.1", "port": AGENT_PORT},
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)  # give uvicorn a moment to bind

    _client_flow(f"http://127.0.0.1:{AGENT_PORT}")

    runner.shutdown(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
