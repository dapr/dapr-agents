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
Configuration for ``ClaudeAgentExecutor``.

Kept free of ``claude_agent_sdk`` imports so the config can be built (and
type-checked) without the optional ``claude`` extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from dapr_agents.agents.executors.binding import ExecutorBinding
from dapr_agents.hooks import BeforeToolHook
from dapr_agents.tool.base import AgentTool

DEFAULT_TOOL_SERVER_NAME = "dapr"


@dataclass(frozen=True)
class ClaudeAgentExecutorConfig:
    """
    Typed configuration for ``ClaudeAgentExecutor``.

    Every field maps onto ``claude_agent_sdk.ClaudeAgentOptions``; the
    executor builds a fresh options object per run so one config can be
    shared by concurrent runs.

    Attributes:
        model: Claude model id (e.g. ``"claude-sonnet-4-5"``). ``None`` uses
            the CLI default.
        system_prompt: System prompt text, or an SDK system-prompt preset
            dict. ``None`` uses the SDK default.
        max_turns: Maximum agentic turns per run.
        max_budget_usd: Optional spend cap per run.
        permission_mode: SDK permission mode (``"default"``,
            ``"acceptEdits"``, ``"plan"``, ``"bypassPermissions"``, ...).
            Passed on every run, including resumes, because the CLI does
            not restore it from the transcript.
        builtin_tools: Built-in Claude Code tools to enable (``Read``,
            ``Bash``, ...). Defaults to none, so the agent can only use the
            dapr-agents tools and MCP servers configured here. ``None``
            keeps the CLI's default tool set.
        allowed_tools: Extra tool names Claude may call without a
            permission prompt (for example ``"mcp__github__get_issue"`` or
            ``"mcp__github"`` for a whole MCP server). Tools from ``tools``
            are allowed automatically. Tools listed here skip the
            ``before_tool_call`` approval gate if the CLI declines to defer
            a call (it only defers a lone tool call per assistant message),
            so do not list tools that need approval.
        disallowed_tools: Tool names Claude must never call.
        tools: dapr-agents tools exposed to Claude through an in-process
            MCP server named ``tool_server_name``. They run inside the
            process (and workflow activity) that drives the executor.
        tool_server_name: Name of that in-process MCP server; Claude sees
            the tools as ``mcp__<tool_server_name>__<tool name>``.
        mcp_servers: Additional MCP server configs (``stdio`` / ``sse`` /
            ``http`` / ``sdk`` dicts as accepted by the SDK).
        before_tool_call: dapr-agents ``before_tool_call`` hooks, evaluated
            in a Claude ``PreToolUse`` hook for every tool call. ``Proceed``
            allows, ``Mutate`` rewrites the arguments, ``Deny`` and ``Skip``
            block the call, and ``RequireApproval`` pauses the run (see
            ``AgentExecutorBase.supports_tool_approval``).
        hooks: Extra raw SDK hooks (``{"PreToolUse": [HookMatcher, ...]}``),
            merged with the executor's own hooks.
        cwd: Working directory for the Claude CLI. The session store key is
            derived from it, so it must be identical on every host that
            resumes a session. ``None`` resolves to the process working
            directory when the executor is created.
        env: Extra environment for the CLI process. Use it to pass
            ``ANTHROPIC_API_KEY`` or ``CLAUDE_CODE_OAUTH_TOKEN`` in pods.
        include_partial_messages: Emit ``text_delta`` events from streamed
            partial messages.
        session_store: A ``claude_agent_sdk`` ``SessionStore`` (for example
            ``DaprSessionStore``) that mirrors transcripts so sessions can
            resume on another host. ``None`` keeps transcripts on local
            disk only.
        setting_sources: Claude settings files to load (``"user"``,
            ``"project"``, ``"local"``). Defaults to none so host settings
            do not leak into agent runs.
        cli_path: Optional path to a ``claude`` CLI binary, overriding the
            one bundled with the SDK wheel.
        extra_options: Additional ``ClaudeAgentOptions`` keyword arguments
            (``thinking``, ``effort``, ``agents``, ``sandbox``, ...). Keys
            that the executor manages (``resume``, ``session_id``,
            ``session_store``, ``hooks``, ``stderr``) are rejected.
    """

    model: Optional[str] = None
    system_prompt: Optional[Any] = None
    max_turns: Optional[int] = None
    max_budget_usd: Optional[float] = None
    permission_mode: Optional[str] = None
    builtin_tools: Optional[Tuple[str, ...]] = ()
    allowed_tools: Tuple[str, ...] = ()
    disallowed_tools: Tuple[str, ...] = ()
    tools: Tuple[AgentTool, ...] = ()
    tool_server_name: str = DEFAULT_TOOL_SERVER_NAME
    mcp_servers: Mapping[str, Any] = field(default_factory=dict)
    before_tool_call: Tuple[BeforeToolHook, ...] = ()
    hooks: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    cwd: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    include_partial_messages: bool = True
    session_store: Optional[Any] = None
    setting_sources: Tuple[str, ...] = ()
    cli_path: Optional[str] = None
    extra_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept any sequence from callers but store immutable tuples.
        for name in _TUPLE_FIELDS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not self.tool_server_name:
            raise ValueError("tool_server_name must not be empty")
        if self.tools and self.tool_server_name in self.mcp_servers:
            raise ValueError(
                f"mcp_servers already defines {self.tool_server_name!r}; "
                "pick another tool_server_name for the dapr-agents tools."
            )
        reserved = RESERVED_OPTION_KEYS.intersection(self.extra_options)
        if reserved:
            raise ValueError(
                "extra_options must not set executor-managed options: "
                + ", ".join(sorted(reserved))
            )

    def tool_names(self) -> Tuple[str, ...]:
        """Names Claude uses for the ``tools`` (``mcp__<server>__<name>``)."""
        prefix = f"mcp__{self.tool_server_name}__"
        return tuple(f"{prefix}{t.name}" for t in self.tools)

    def bound_to(self, binding: ExecutorBinding) -> "ClaudeAgentExecutorConfig":
        """
        Return a copy filled in from the hosting agent's ``binding``.

        Explicit settings win: ``system_prompt``, ``max_turns`` and
        ``session_store`` come from the binding only when unset here. The
        binding's tools and ``before_tool_call`` hooks are added after this
        config's own, skipping tools whose name is already configured.
        """
        own_names = {t.name for t in self.tools}
        extra_tools = tuple(t for t in binding.tools if t.name not in own_names)
        extra_hooks = tuple(
            h for h in binding.before_tool_call if h not in self.before_tool_call
        )
        return replace(
            self,
            system_prompt=_first_set(self.system_prompt, binding.system_prompt),
            max_turns=_first_set(self.max_turns, binding.max_iterations),
            tools=self.tools + extra_tools,
            before_tool_call=self.before_tool_call + extra_hooks,
            session_store=_first_set(self.session_store, binding.session_store),
        )


def _first_set(value: Any, fallback: Any) -> Any:
    return value if value is not None else fallback


_TUPLE_FIELDS = (
    "builtin_tools",
    "allowed_tools",
    "disallowed_tools",
    "tools",
    "before_tool_call",
    "setting_sources",
)

RESERVED_OPTION_KEYS = frozenset(
    {
        "resume",
        "session_id",
        "session_store",
        "hooks",
        "stderr",
        "continue_conversation",
        "fork_session",
        "include_partial_messages",
    }
)
