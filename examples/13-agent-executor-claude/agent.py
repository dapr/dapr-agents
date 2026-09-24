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
Builds a DurableAgent whose reasoning loop runs in the Claude Agent SDK.

* ``get_weather`` is a normal dapr-agents tool that Claude may call freely.
* ``transfer_money`` is gated by a ``before_tool_call`` hook that returns
  ``RequireApproval``: the run pauses durably until a human sends a decision.
* The agent's profile, tools, hooks and ``max_iterations`` are handed to
  Claude automatically: DurableAgent binds the executor to them before every
  run. Claude session transcripts are mirrored into the agent's state store
  (``agentstatestore``), so a follow-up turn (or a retried activity, or
  another pod) resumes the same Claude session.
"""

import os
from pathlib import Path

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentExecutionConfig, AgentStateConfig
from dapr_agents.agents.executors import (
    ClaudeAgentExecutor,
    ClaudeAgentExecutorConfig,
)
from dapr_agents.hooks import (
    HookDecision,
    Hooks,
    Proceed,
    RequireApproval,
    ToolHookContext,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService

# Dapr state store component (resources/statestore.yaml) used for the
# agent's own state and for the Claude session transcripts.
STATE_STORE = "agentstatestore"
DEFAULT_MODEL = "claude-haiku-4-5"
APPROVAL_TIMEOUT_SECONDS = 300
# Caps Claude's agentic turns per run (and the number of approval rounds).
MAX_ITERATIONS = 6

INSTRUCTIONS = (
    "Be concise and use the available tools to answer.",
    # Claude defers a tool call cleanly only when it is the sole call in its
    # message, so ask for one call at a time.
    "Call at most one tool at a time.",
    "Never retry a transfer that a human rejected.",
)

WEATHER = {
    "paris": "sunny, 22°C",
    "london": "light rain, 15°C",
    "tokyo": "cloudy, 18°C",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    forecast = WEATHER.get(city.strip().lower(), "clear skies, 20°C")
    return f"The weather in {city} is {forecast}."


@tool
def transfer_money(to: str, amount: float) -> str:
    """Transfer an amount of US dollars from the user's account to a recipient."""
    # A real implementation would call a payments API here. It only runs
    # after a human approved this exact call.
    return f"Transferred ${amount:.2f} to {to}."


def require_approval_for_transfers(ctx: ToolHookContext) -> HookDecision:
    """Pause the run for a human decision before any money moves."""
    if ctx.step_name == "transfer_money":
        return RequireApproval(
            timeout_seconds=APPROVAL_TIMEOUT_SECONDS,
            instructions=(
                f"Approve sending ${ctx.payload.get('amount')} "
                f"to {ctx.payload.get('to')}?"
            ),
        )
    return Proceed()


def build_executor() -> ClaudeAgentExecutor:
    """Create the Claude executor.

    Only Claude-specific settings live here. The system prompt, tools,
    hooks, turn limit and the Dapr session store come from the DurableAgent.
    """
    return ClaudeAgentExecutor(
        ClaudeAgentExecutorConfig(
            model=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
            max_budget_usd=0.50,
            # The session store key is derived from cwd, so keep it identical
            # on every host that resumes a session.
            cwd=str(Path(__file__).resolve().parent),
        )
    )


def build_agent() -> DurableAgent:
    """Create the DurableAgent that delegates each run to Claude."""
    return DurableAgent(
        name="ClaudeAssistant",
        role="Personal Assistant",
        goal="Answer questions and move money only with human approval.",
        instructions=list(INSTRUCTIONS),
        executor=build_executor(),
        tools=[get_weather, transfer_money],
        hooks=Hooks(before_tool_call=[require_approval_for_transfers]),
        state=AgentStateConfig(store=StateStoreService(store_name=STATE_STORE)),
        execution=AgentExecutionConfig(max_iterations=MAX_ITERATIONS),
    )
