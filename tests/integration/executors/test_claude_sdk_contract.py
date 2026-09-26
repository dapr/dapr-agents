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

"""Canary for the Claude CLI behaviour ClaudeAgentExecutor relies on.

A ``PreToolUse`` hook answering ``"defer"`` must end the run with the
pending call, and a resume must replay that call through ``PreToolUse``.
Runs against the real CLI when ``ANTHROPIC_API_KEY`` is set, or with
``DAPR_AGENTS_CLAUDE_LIVE=1`` to use a local Claude login.
"""

import os

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import InMemorySessionStore  # noqa: E402

from dapr_agents.agents.executors import (  # noqa: E402
    CONTEXT_TOOL_DECISIONS,
    ClaudeAgentExecutor,
    ClaudeAgentExecutorConfig,
    ToolCallDecision,
    arguments_digest,
)
from dapr_agents.agents.executors.claude import STALE_CALL_REASON  # noqa: E402
from dapr_agents.hooks import Proceed, RequireApproval  # noqa: E402
from dapr_agents.tool import tool  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("DAPR_AGENTS_CLAUDE_LIVE")),
        reason="needs ANTHROPIC_API_KEY or DAPR_AGENTS_CLAUDE_LIVE=1",
    ),
]

TRANSFERS = []


@tool
def transfer_money(to: str, amount: int) -> str:
    """Transfer an amount of US dollars to a recipient."""
    TRANSFERS.append((to, amount))
    return f"Transferred ${amount} to {to}."


def _gate(ctx):
    return RequireApproval() if ctx.step_name == "transfer_money" else Proceed()


def _executor(tmp_path, store):
    return ClaudeAgentExecutor(
        ClaudeAgentExecutorConfig(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5"),
            system_prompt=(
                "Call transfer_money directly when asked, one tool at a time."
                " Never retry a transfer that was rejected."
            ),
            max_turns=3,
            max_budget_usd=0.20,
            tools=[transfer_money],
            before_tool_call=[_gate],
            session_store=store,
            cwd=str(tmp_path),
        )
    )


async def _run(executor, prompt, **kwargs):
    return [event async for event in executor.run(prompt, **kwargs)]


def _decision(paused, approved):
    content = paused.content
    decision = ToolCallDecision(
        tool_call_id=content["tool_call_id"],
        approved=approved,
        reason=None if approved else "rejected by the canary",
        arguments_digest=arguments_digest(content["arguments"]),
    )
    return {CONTEXT_TOOL_DECISIONS: {decision.tool_call_id: decision.to_dict()}}


async def _pause(executor):
    events = await _run(executor, "Send $5 to Alice.")
    paused = events[-1]
    assert paused.type == "paused", events[-1]
    assert paused.content["name"] == "transfer_money"
    return paused


@pytest.mark.parametrize("approved", [True, False])
async def test_deferred_call_resumes_with_the_decision(tmp_path, approved):
    TRANSFERS.clear()
    executor = _executor(tmp_path, InMemorySessionStore())
    paused = await _pause(executor)
    assert TRANSFERS == []

    events = await _run(
        executor,
        "",
        session_id=paused.session_id,
        context=_decision(paused, approved),
    )

    assert TRANSFERS == ([("Alice", 5)] if approved else [])
    if approved:
        assert events[-1].type == "complete", events[-1]
    elif events[-1].type == "paused":
        # Claude may ask again; a new call needs its own approval.
        assert events[-1].content["tool_call_id"] != paused.content["tool_call_id"]


async def test_stale_deferred_call_is_denied_before_the_next_task(tmp_path):
    TRANSFERS.clear()
    executor = _executor(tmp_path, InMemorySessionStore())
    paused = await _pause(executor)

    events = await _run(
        executor, "Reply with the word: ok", session_id=paused.session_id
    )

    assert events[-1].type == "complete", events[-1]
    assert TRANSFERS == []
    transcript = await executor.get_session(paused.session_id)
    assert STALE_CALL_REASON in str(transcript)
