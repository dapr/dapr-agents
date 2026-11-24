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
HITL example using the built-in ``ask_user`` tool.

The LLM can call ``ask_user`` mid-turn to ask the human a question before
finishing its turn. This suspends the workflow until the human answers, then
resumes it.

Agent side : enabling ``ask_user`` suspends the workflow and emits a
             ``USER_INPUT_REQUESTED`` chunk on the run's stream.
Client side: start the run via ``POST /agent/run/stream``, watch the stream
             for that chunk, and answer via ``POST /agent/run/input``.

If the client's connection drops before it answers, the question isn't lost:
``GET /agent/instances/{id}`` returns it under ``pending_inputs``, so the
client can recover the request and still reply through ``/input``. This
script simulates that: it reads the question, closes the stream, then
recovers and answers it through the status endpoint instead.

This script runs both the agent and the client in the same process for
simplicity.
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
    """Run the chat-client side of the demo."""

    asyncio.run(_async_client_flow(base_url))


async def _async_client_flow(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start the run and read only the first question the agent asks.
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

        # Simulate the chat client disconnecting (page reload, lost
        # connection) before answering.
        print("[chat client] stream closed, simulating a dropped connection")
        await asyncio.sleep(1)

        status_url = f"{base_url}/agent/instances/{target_instance_id}"
        status = (await client.get(status_url)).json()
        pending = status.get("pending_inputs") or []
        if not pending:
            print("No pending input recovered from the status endpoint.")
            return
        recovered = pending[0]
        print(f"[chat client] recovered question: {recovered['question']!r}")

        # Answer using the recovered request_id / target_instance_id.
        print(f"[chat client] answering: {ANSWER}\n")
        await client.post(
            f"{base_url}/agent/run/input",
            json={
                "request_id": recovered["request_id"],
                "target_instance_id": recovered["target_instance_id"],
                "answer": ANSWER,
            },
        )

        # Poll the same instance until the turn completes.
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
