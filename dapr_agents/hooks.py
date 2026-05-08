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
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class HookContext:
    """all the information available to a hook when a step is about to run."""

    step_name: str
    """name of the tool about to run, or 'llm' for llm calls."""

    step_kind: str
    """'tool' for tool calls, 'llm' for llm calls."""

    source: str
    """where this tool came from: 'local', 'mcp', 'openapi', etc."""

    payload: Dict[str, Any]
    """arguments the llm wants to pass to the tool (or llm call params)."""

    tool_call_id: str = ""
    """llm-assigned id for this specific call. empty for llm-level hooks."""


class HookDecision:
    """base class — you never instantiate this directly, use the subclasses."""

    pass


@dataclass
class Proceed(HookDecision):
    """run the step normally. returning None from a hook coerces to this."""

    pass


@dataclass
class Skip(HookDecision):
    """
    skip execution entirely and use `result` as the step output instead.
    useful for returning cached results or safe defaults on policy checks.
    """

    result: Any = None


@dataclass
class Modify(HookDecision):
    """
    run the step but replace the incoming arguments with `payload` first.
    for tool calls, `payload` is the new arguments dict passed to the tool.
    """

    payload: Optional[Dict[str, Any]] = None


@dataclass
class RequireApproval(HookDecision):
    """
    pause the workflow and wait for a human decision before running the step.
    if no human responds within `timeout_seconds`, the step is auto-denied.

    this is the hook decision that drives the HITL flow — it triggers the same
    publish → wait_for_external_event → timer-race plumbing as before, but now
    any tool (local, mcp, openapi) can trigger it, not just ones with a decorator.
    """

    timeout_seconds: Optional[int] = None
    """per-call timeout override. falls back to AgentApprovalConfig.default_timeout_seconds."""

    instructions: Optional[str] = None
    """message shown to the approver explaining what needs a decision."""

    reason: Optional[str] = None
    """optional context about why this step needs a human decision."""


@dataclass
class Deny(HookDecision):
    """
    block the step without involving a human. the workflow synthesizes a
    ToolMessage so the llm knows the call was blocked and can respond.
    """

    reason: Optional[str] = None


BeforeHook = Callable[[HookContext], Optional[HookDecision]]
AfterHook = Callable[[HookContext, Any], Optional[HookDecision]]


@dataclass
class Hooks:
    """
    container for all hook callbacks you want to register on a DurableAgent.
    each slot holds a list of callables so multiple hooks can be chained.

    example::

        from dapr_agents.hooks import Hooks, HookContext, HookDecision
        from dapr_agents.hooks import Proceed, RequireApproval, Deny

        def before_tool(ctx: HookContext) -> HookDecision:
            # gate any mcp delete_ call through human approval
            if ctx.source == "mcp" and ctx.step_name.startswith("delete_"):
                return RequireApproval(
                    timeout_seconds=3600,
                    instructions=f"confirm deletion: {ctx.payload}",
                )
            # outright block schema-altering calls
            if ctx.step_name == "drop_table":
                return Deny(reason="schema changes go through dba review")
            return Proceed()

        agent = DurableAgent(
            ...,
            hooks=Hooks(before_tool_call=[before_tool]),
        )
    """

    before_tool_call: List[BeforeHook] = field(default_factory=list)
    """called before every tool dispatch. return a HookDecision to control execution."""

    after_tool_call: List[AfterHook] = field(default_factory=list)
    """called after a tool completes. return Modify(result=...) to replace the output."""

    before_llm_call: List[BeforeHook] = field(default_factory=list)
    """called before every llm call."""

    after_llm_call: List[AfterHook] = field(default_factory=list)
    """called after every llm response."""
