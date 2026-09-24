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
``AgentExecutorBase`` implementation backed by the Claude Agent SDK.

``ClaudeAgentExecutor`` runs Claude Code's agent loop (via
``claude_agent_sdk.ClaudeSDKClient``) and translates its message stream
into ``AgentEvent`` values. Requires the optional ``claude`` extra::

    pip install "dapr-agents[claude]"

Sessions: every run uses a UUID session id. When a ``session_store`` is
configured the transcript is mirrored into it, and a later run with the
same ``session_id`` resumes from the store, on any host, as long as the
executor's ``cwd`` is the same (the store key derives from it).

Tool approval: ``before_tool_call`` hooks that return ``RequireApproval``
defer the call. The run then ends with a ``paused`` event and is resumed
by passing ``context[CONTEXT_TOOL_DECISIONS]`` (see
``dapr_agents.agents.executors.event``).
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Dict, List, Mapping, Optional, cast

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ClaudeSDKError,
        CLINotFoundError,
        HookMatcher,
        get_session_info,
        get_session_info_from_store,
        get_session_messages,
        get_session_messages_from_store,
        project_key_for_directory,
    )
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "ClaudeAgentExecutor requires the Claude Agent SDK. "
        'Install it with: pip install "dapr-agents[claude]"'
    ) from exc

from dapr_agents.agents.executors.base import AgentExecutorBase
from dapr_agents.agents.executors.binding import ExecutorBinding
from dapr_agents.agents.executors.claude_config import ClaudeAgentExecutorConfig
from dapr_agents.agents.executors.claude_events import ClaudeEventMapper
from dapr_agents.agents.executors.claude_tools import ToolGate, build_tool_server
from dapr_agents.agents.executors.claude_transcript import (
    final_assistant_text,
    last_total_cost,
    pending_deferred_call,
)
from dapr_agents.agents.executors.event import (
    EVENT_COMPLETE,
    EVENT_ERROR,
    AgentEvent,
    ToolCallDecision,
    tool_decisions_from_context,
)

logger = logging.getLogger(__name__)

# Namespace for deriving UUID session ids from caller ids that are not UUIDs
# (the Claude CLI only accepts UUIDs).
_SESSION_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "dapr-agents/claude-session")
_STDERR_TAIL_LINES = 20
_CLI_MISSING_HINT = (
    "Claude Code CLI not found. The claude-agent-sdk wheel bundles it on "
    "macOS, glibc Linux and Windows; on other platforms (e.g. Alpine/musl) "
    "install the `claude` CLI on PATH or set cli_path."
)


def session_uuid(session_id: str) -> str:
    """Return ``session_id`` if it is a UUID, else a stable UUID derived from it."""
    try:
        return str(uuid.UUID(session_id))
    except ValueError:
        return str(uuid.uuid5(_SESSION_NAMESPACE, session_id))


@dataclass(frozen=True)
class _RunPlan:
    """How a single ``run`` call talks to the CLI."""

    session_id: str
    resume: bool = False
    send_prompt: bool = False
    prior_cost_usd: Optional[float] = None
    replay_text: Optional[str] = None
    error: Optional[str] = None


class ClaudeAgentExecutor(AgentExecutorBase):
    """
    Stateful executor running the Claude Agent SDK agent loop.

    Args:
        config: Typed executor configuration. Defaults to
            ``ClaudeAgentExecutorConfig()`` (CLI default model, no built-in
            tools, no settings files, local-disk sessions only).

    Inside a ``DurableAgent`` the executor is bound (see ``bind``) to the
    agent's system prompt, ``max_iterations``, tools, ``before_tool_call``
    hooks and a ``DaprSessionStore`` on the agent's state store, for every
    setting left unset in ``config``.

    Example:
        executor = ClaudeAgentExecutor(
            ClaudeAgentExecutorConfig(model="claude-sonnet-4-5")
        )
        agent = DurableAgent(
            name="claude",
            role="Weather assistant",
            executor=executor,
            tools=[get_weather],
            state=AgentStateConfig(store=StateStoreService(store_name="statestore")),
        )
    """

    supports_tool_approval = True

    def __init__(self, config: Optional[ClaudeAgentExecutorConfig] = None) -> None:
        resolved = config or ClaudeAgentExecutorConfig()
        if resolved.cwd is None:
            resolved = dataclasses.replace(resolved, cwd=os.getcwd())
        self._config = resolved

    @property
    def config(self) -> ClaudeAgentExecutorConfig:
        """The executor configuration (``cwd`` always resolved)."""
        return self._config

    @property
    def cwd(self) -> str:
        """Working directory passed to the Claude CLI."""
        return str(self._config.cwd)

    @property
    def project_key(self) -> str:
        """``SessionStore`` project key derived from ``cwd``."""
        return project_key_for_directory(self.cwd)

    # ------------------------------------------------------------------
    # AgentExecutorBase
    # ------------------------------------------------------------------

    def bind(self, binding: ExecutorBinding) -> "ClaudeAgentExecutor":
        """
        Return a copy that uses the hosting agent's prompt, tools and store.

        Settings made explicitly on this executor's config win; see
        ``ClaudeAgentExecutorConfig.bound_to``.
        """
        return ClaudeAgentExecutor(self._config.bound_to(binding))

    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        Run one Claude turn, or resume a paused one.

        Args:
            prompt: Task text. Ignored when resuming a paused run.
            session_id: Session to resume; a fresh UUID session when
                ``None``. Non-UUID ids are mapped to a stable UUID, which
                is what events report as ``session_id``.
            context: Optional ``CONTEXT_TOOL_DECISIONS`` mapping to resume a
                paused run. Other keys are ignored.

        Yields:
            ``AgentEvent`` values ending in ``complete``, ``paused`` or
            ``error``. SDK failures become ``error`` events, not exceptions.
        """
        decisions = tool_decisions_from_context(context)
        try:
            plan = await self._plan(session_id, decisions)
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            logger.exception("Failed to prepare Claude session %s", session_id)
            plan = _RunPlan(session_id=session_id or "", error=str(exc))
        if plan.error is not None or plan.replay_text is not None:
            yield self._short_circuit(plan)
            return
        async for event in self._execute(prompt, plan, decisions):
            yield event

    async def _execute(
        self,
        prompt: str,
        plan: _RunPlan,
        decisions: Mapping[str, ToolCallDecision],
    ) -> AsyncGenerator[AgentEvent, None]:
        gate = ToolGate(
            hooks=self._config.before_tool_call,
            decisions=decisions,
            tool_prefix=f"mcp__{self._config.tool_server_name}__",
        )
        mapper = ClaudeEventMapper(
            session_id=plan.session_id,
            display_name=gate.display_name,
            approval_for=lambda call_id: _approval_dict(gate, call_id),
            include_text_deltas=self._config.include_partial_messages,
            prior_cost_usd=plan.prior_cost_usd,
        )
        stderr_tail: Deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        failure: Optional[AgentEvent] = None
        try:
            options = self._options(plan, gate, stderr_tail)
            async with ClaudeSDKClient(options=options) as client:
                if plan.send_prompt:
                    await client.query(prompt)
                # Drain to the end: the session store gets its final append
                # while the client shuts down, so terminal events are only
                # yielded after the ``async with`` block exits.
                async for message in client.receive_response():
                    for event in mapper.map(message):
                        yield event
        except CLINotFoundError as exc:
            failure = mapper.error(f"{_CLI_MISSING_HINT} ({exc})")
        except Exception as exc:  # noqa: BLE001 - SDK, CLI and store failures
            # An is_error ResultMessage is followed by a ResultError; the
            # mapper already built the richer error event from the result.
            if mapper.terminal is None:
                logger.warning(
                    "Claude run failed (session=%s): %s", plan.session_id, exc
                )
                failure = mapper.error(_describe_failure(exc, stderr_tail))
            elif not isinstance(exc, ClaudeSDKError):
                logger.exception(
                    "Claude run failed after its result (session=%s)", plan.session_id
                )
            else:
                logger.debug("Claude CLI exited after terminal result: %s", exc)

        yield (
            mapper.terminal
            or failure
            or mapper.error("Claude run ended without a result message")
        )

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Return ``{"session_id", "messages", "metadata"}`` for a session.

        Reads from the configured session store, or from the CLI's local
        transcripts when no store is configured. ``messages`` holds the
        user/assistant transcript messages (``type``, ``uuid``, ``message``).
        """
        sid = session_uuid(session_id)
        store = self._config.session_store
        if store is not None:
            info = await get_session_info_from_store(store, sid, directory=self.cwd)
            messages = await get_session_messages_from_store(
                store, sid, directory=self.cwd
            )
        else:
            info = await asyncio.to_thread(get_session_info, sid, directory=self.cwd)
            messages = await asyncio.to_thread(
                get_session_messages, sid, directory=self.cwd
            )
        if info is None and not messages:
            return None
        return {
            "session_id": sid,
            "messages": [
                {"type": m.type, "uuid": m.uuid, "message": m.message} for m in messages
            ],
            "metadata": dataclasses.asdict(info) if info is not None else {},
        }

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _load_transcript(self, sid: str) -> Optional[List[Any]]:
        store = self._config.session_store
        if store is None:
            return None
        key = {"project_key": self.project_key, "session_id": sid}
        return await store.load(key)

    async def _session_exists(self, sid: str, entries: Optional[List[Any]]) -> bool:
        if self._config.session_store is not None:
            return bool(entries)
        info = await asyncio.to_thread(get_session_info, sid, directory=self.cwd)
        return info is not None

    async def _plan(
        self, session_id: Optional[str], decisions: Mapping[str, ToolCallDecision]
    ) -> _RunPlan:
        if session_id is None:
            if decisions:
                return _RunPlan(
                    session_id="", error="Cannot resume without a session_id"
                )
            return _RunPlan(session_id=str(uuid.uuid4()), send_prompt=True)

        sid = session_uuid(session_id)
        entries = await self._load_transcript(sid)
        exists = await self._session_exists(sid, entries)
        prior_cost = last_total_cost(entries) if entries else None
        if not decisions:
            return _RunPlan(
                session_id=sid,
                resume=exists,
                send_prompt=True,
                prior_cost_usd=prior_cost,
            )
        if not exists:
            return _RunPlan(session_id=sid, error=f"No Claude session {sid} to resume")
        if entries is not None and pending_deferred_call(entries) is None:
            # Nothing is waiting for a decision: a previous attempt already
            # resumed and finished (e.g. the activity is being retried), so
            # replay its answer instead of re-running the task.
            text = final_assistant_text(entries)
            if text is None:
                return _RunPlan(
                    session_id=sid,
                    error=f"Claude session {sid} has no paused tool call",
                )
            return _RunPlan(session_id=sid, replay_text=text)
        # Resume without a prompt: the CLI re-runs the deferred call through
        # PreToolUse, where the ToolGate applies the decision.
        return _RunPlan(session_id=sid, resume=True, prior_cost_usd=prior_cost)

    @staticmethod
    def _short_circuit(plan: _RunPlan) -> AgentEvent:
        if plan.error is not None:
            return AgentEvent(
                type=EVENT_ERROR, content=plan.error, session_id=plan.session_id or None
            )
        return AgentEvent(
            type=EVENT_COMPLETE,
            content={"role": "assistant", "content": plan.replay_text},
            session_id=plan.session_id,
            metadata={"replayed": True, "cost_usd": 0.0},
        )

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def _hooks(self, gate: ToolGate) -> Optional[Dict[str, List[Any]]]:
        hooks: Dict[str, List[Any]] = {
            event: list(matchers) for event, matchers in self._config.hooks.items()
        }
        if gate.enabled:
            hooks["PreToolUse"] = [
                HookMatcher(matcher=None, hooks=[cast(Any, gate)]),
                *hooks.get("PreToolUse", []),
            ]
        return hooks or None

    def _mcp_servers(self) -> Dict[str, Any]:
        servers = dict(self._config.mcp_servers)
        if self._config.tools:
            servers[self._config.tool_server_name] = build_tool_server(
                self._config.tool_server_name, self._config.tools
            )
        return servers

    def _options(
        self, plan: _RunPlan, gate: ToolGate, stderr_tail: Deque[str]
    ) -> ClaudeAgentOptions:
        cfg = self._config
        os.makedirs(self.cwd, exist_ok=True)

        def on_stderr(line: str) -> None:
            stderr_tail.append(line)
            logger.debug("claude[%s]: %s", plan.session_id, line.rstrip())

        builtin = list(cfg.builtin_tools) if cfg.builtin_tools is not None else None
        return ClaudeAgentOptions(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            max_turns=cfg.max_turns,
            max_budget_usd=cfg.max_budget_usd,
            permission_mode=cast(Any, cfg.permission_mode),
            tools=builtin,
            # With the gate active, dapr-agents tools are allowed by the gate
            # itself so a gated call never runs on a static allow rule.
            allowed_tools=[
                *(() if gate.enabled else cfg.tool_names()),
                *cfg.allowed_tools,
            ],
            disallowed_tools=list(cfg.disallowed_tools),
            mcp_servers=self._mcp_servers(),
            hooks=cast(Any, self._hooks(gate)),
            cwd=self.cwd,
            env=cfg.cli_env(),
            strict_mcp_config=cfg.isolate_host_config,
            include_partial_messages=cfg.include_partial_messages,
            session_store=cfg.session_store,
            setting_sources=cast(Any, list(cfg.setting_sources)),
            cli_path=cfg.cli_path,
            stderr=on_stderr,
            resume=plan.session_id if plan.resume else None,
            session_id=None if plan.resume else plan.session_id,
            **dict(cfg.extra_options),
        )


def _approval_dict(gate: ToolGate, call_id: str) -> Optional[Dict[str, Any]]:
    approval = gate.deferred_approval(call_id)
    return approval.to_dict() if approval is not None else None


def _describe_failure(exc: Exception, stderr_tail: Deque[str]) -> str:
    detail = " | ".join(line.strip() for line in stderr_tail if line.strip())
    message = f"Claude Agent SDK error: {exc}"
    return f"{message} (stderr: {detail})" if detail else message
