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

"""Chainlit entrypoint for the expert agent."""

import json
import uuid
from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from dapr_agents import AgentRunner, DurableAgent

from agent import build_agent


load_dotenv()


# Lazily initialized on first chat session so the Dapr sidecar is ready before
# the agent's workflow runtime tries to register itself.
_agent: Optional[DurableAgent] = None
_runner: Optional[AgentRunner] = None

# Key under which we stash the per-chat workflow instance id in cl.user_session.
_INSTANCE_ID_KEY = "agent_instance_id"


def _get_agent_and_runner() -> tuple[DurableAgent, AgentRunner]:
    global _agent, _runner
    if _agent is None:
        _agent = build_agent()
        _runner = AgentRunner()
        _runner.workflow(_agent)
    return _agent, _runner


def _extract_content(result: str | None) -> str:
    """Pull the assistant content out of a serialized workflow result."""
    if not result:
        return ""
    try:
        parsed = json.loads(result)
        return parsed.get("content", result)
    except (json.JSONDecodeError, AttributeError):
        return result


@cl.on_chat_start
async def start() -> None:
    _get_agent_and_runner()
    # One workflow instance per chat session — all messages in this session
    # use the same instance_id so conversation memory accumulates. Without
    # this, every message would spawn a fresh workflow and the agent would
    # forget the previous turns.
    cl.user_session.set(_INSTANCE_ID_KEY, f"chat-{uuid.uuid4()}")
    await cl.Message(
        content=(
            "Hi! Ask me anything — I'll fetch fresh web context for every "
            "question via a `before_llm_call` hook backed by Tavily."
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent, runner = _get_agent_and_runner()
    instance_id = cl.user_session.get(_INSTANCE_ID_KEY)
    if instance_id is None:
        # Defensive: if the chat-start hook didn't run (e.g. session restored
        # mid-conversation), generate one on the fly.
        instance_id = f"chat-{uuid.uuid4()}"
        cl.user_session.set(_INSTANCE_ID_KEY, instance_id)
    result = await runner.run(agent, {"task": message.content}, instance_id=instance_id)
    await cl.Message(content=_extract_content(result)).send()
