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
``DurableAgent`` support for ``AgentExecutorBase`` runtimes.

``DurableExecutorMixin`` holds the executor branch of ``agent_workflow``
and the ``run_executor`` activity:

* Before every run the executor is bound (``AgentExecutorBase.bind``) to the
  agent's system prompt, ``max_iterations``, tools (including registry
  agents-as-tools and Dapr ``MCPServer`` tools, bridged so they can run in
  the activity), ``before_tool_call`` hooks and a ``DaprSessionStore`` on
  the agent's state store.
* A run that pauses on a tool call awaiting approval returns a
  ``PausedExecutorRun``. The workflow body then runs the normal approval
  flow (``_await_approval``: publish, wait for the external event or the
  timeout) and calls ``run_executor`` again with the decision under
  ``context[CONTEXT_TOOL_DECISIONS]``, repeating until the run completes.
  A rejection passes the approver's reason on to the executor.
* After ``execution.max_approval_rounds`` (default ``max_iterations``)
  approval rounds, a further paused call is rejected without asking a human
  and the run gets one last chance to finish. A call left paused after that,
  or by a workflow terminated while waiting, is rejected by the executor on
  the next run of the same session, before the new task is sent.

Delivery semantics: ``run_executor`` is an activity, so it runs at least
once. A retried attempt resumes the executor session saved by the previous
attempt (the task prompt is sent again), and tools that ran inside the
failed attempt may run again. Failures an executor marks as not retryable
(``METADATA_RETRYABLE``) fail the workflow instead of being retried. Workflow-backed tools reuse the child
workflow of the earlier attempt (their instance ids are deterministic).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import dapr.ext.workflow as wf

from dapr_agents.agents.executor_run import (
    ExecutorRunRecorder,
    PausedExecutorRun,
    executor_failure,
)
from dapr_agents.agents.executors import (
    CONTEXT_TOOL_DECISIONS,
    AgentExecutorBase,
    arguments_digest,
    DaprSessionStore,
    DaprSessionStoreConfig,
    ToolCallDecision,
)
from dapr_agents.agents.executors.binding import ExecutorBinding
from dapr_agents.agents.executors.observer import ExecutorRunInfo, ExecutorRunObserver
from dapr_agents.tool.workflow.activity_bridge import (
    ActivityToolContext,
    bridge_workflow_tools,
)
from dapr_agents.types import AgentError

logger = logging.getLogger(__name__)

_DENIED_REASON = "approval was not granted or timed out"
_LIMIT_REASON = (
    "the approval limit for this task was reached; do not request more tool "
    "calls that need approval"
)
_APPROVAL_LIMIT_MESSAGE = (
    "I reached the maximum number of tool approvals for this task before I "
    "could finish. Please try again with a narrower request."
)


def _resume_input(
    first_input: Dict[str, Any],
    paused: PausedExecutorRun,
    approved: bool,
    round_: int,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``run_executor`` input that resumes ``paused`` with a decision."""
    decision = ToolCallDecision(
        tool_call_id=paused.tool_call_id,
        approved=approved,
        reason=None if approved else (reason or _DENIED_REASON),
        arguments_digest=arguments_digest(paused.arguments),
    )
    context = dict(first_input.get("context") or {})
    context[CONTEXT_TOOL_DECISIONS] = {paused.tool_call_id: decision.to_dict()}
    resumed: Dict[str, Any] = {
        "task": None,
        "instance_id": first_input["instance_id"],
        "source": first_input.get("source"),
        "context": context,
        "round": round_,
    }
    session_id = paused.session_id or first_input.get("session_id")
    if session_id:
        resumed["session_id"] = session_id
    return resumed


class DurableExecutorMixin:
    """Executor branch of ``DurableAgent`` (see module docs)."""

    # ------------------------------------------------------------------
    # Workflow body
    # ------------------------------------------------------------------
    def _executor_workflow(
        self,
        ctx: wf.DaprWorkflowContext,
        *,
        task: Optional[str],
        source: str,
        message: Dict[str, Any],
    ):
        """
        Drive ``run_executor`` until the run completes; ``yield from`` it.

        Every round is a recorded activity and approvals use deterministic
        ids, so the loop is replay-safe. Approval rounds are capped at
        ``execution.max_approval_rounds``; past the cap, one more paused call
        is rejected automatically.

        Returns:
            The final assistant message dict.
        """
        if not ctx.is_replaying:
            logger.debug(
                "Agent %s delegating to executor %s (instance=%s)",
                self.name,
                type(self.executor).__name__,
                ctx.instance_id,
            )
        first_input: Dict[str, Any] = {
            "task": task,
            "instance_id": ctx.instance_id,
            "source": source,
        }
        # Caller-supplied session_id resumes a prior executor session;
        # omitting it lets the executor auto-assign per its contract.
        if message.get("session_id"):
            first_input["session_id"] = message["session_id"]
        if isinstance(message.get("context"), dict):
            first_input["context"] = message["context"]

        max_approvals = (
            self.execution.max_approval_rounds
            if self.execution.max_approval_rounds is not None
            else self.execution.max_iterations
        )
        payload = first_input
        round_ = 0
        while True:
            result = yield ctx.call_activity(
                self._activity_name(self.run_executor),
                input=payload,
                retry_policy=self._retry_policy,
            )
            error = executor_failure(result)
            if error is not None:
                raise AgentError(error)
            paused = PausedExecutorRun.from_activity_result(result)
            if paused is None:
                return result
            if round_ > max_approvals:
                break
            round_ += 1
            if round_ > max_approvals:
                self._log_approval_limit(ctx, max_approvals)
                payload = _resume_input(
                    first_input, paused, False, round_, _LIMIT_REASON
                )
                continue
            approved, reason = yield from self._await_approval(
                ctx,
                ctx.instance_id,
                paused.tool_call(),
                paused.require_approval(),
                source=paused.source,
            )
            payload = _resume_input(first_input, paused, approved, round_, reason)

        return {"role": "assistant", "content": _APPROVAL_LIMIT_MESSAGE}

    def _log_approval_limit(self, ctx: wf.DaprWorkflowContext, limit: int) -> None:
        if not ctx.is_replaying:
            logger.warning(
                "Agent %s hit max approval rounds (%d); rejecting further "
                "calls (instance=%s)",
                self.name,
                limit,
                ctx.instance_id,
            )

    # ------------------------------------------------------------------
    # Activity
    # ------------------------------------------------------------------
    def run_executor(
        self,
        ctx: wf.WorkflowActivityContext,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Drive an ``AgentExecutorBase`` run to its end and return its result.

        The executor owns the full tool/reasoning loop; this activity binds
        it to the agent, consumes its event stream, mirrors events into
        Dapr state, streams text deltas and checkpoints on ``session``
        events.

        Args:
            payload: Required keys ``task``, ``instance_id``, ``source``;
                optional ``session_id`` (resume a prior executor session),
                ``context`` (provider-specific extras, including tool-call
                decisions when resuming a paused run) and ``round`` (0 for
                the first run of a workflow, then one per approval resume).

        Returns:
            The final assistant message dict, or a paused-run dict (see
            ``PausedExecutorRun``) when the run awaits a tool approval.

        Raises:
            AgentError: If the executor is not configured, yields an
                ``error`` event, or ends without a terminal event.
        """
        if self.executor is None:  # Defensive; agent_workflow guards this.
            raise AgentError(
                "run_executor called on an agent without an AgentExecutorBase."
            )

        return self._run_asyncio_task(self._consume_executor(payload))

    async def _consume_executor(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consume one executor run; see ``ExecutorRunRecorder`` for the events.

        State is persisted in a ``finally`` block so every exit path
        (success, pause, ``error`` event, executor exception, missing
        terminal event) flushes the accumulated entry exactly once.
        """
        instance_id: str = payload["instance_id"]
        task: Optional[str] = payload.get("task")
        round_ = int(payload.get("round", 0) or 0)
        entry = self._infra.get_state(instance_id)
        self._record_executor_task(instance_id, task, payload.get("source"), entry)

        executor = self._bind_executor(instance_id, round_, entry)
        # Session id priority: caller-supplied, then the id a previous
        # attempt persisted (retry-safe), else the executor assigns one.
        session_id = payload.get("session_id") or getattr(entry, "session_id", None)
        emitter = self._executor_stream_emitter(instance_id, round_, entry)
        observer = self._executor_observer(
            ExecutorRunInfo(
                agent_name=self.name,
                executor_type=type(executor).__name__,
                instance_id=instance_id,
                round=round_,
                session_id=session_id,
                model=_executor_model(executor),
            )
        )
        recorder = ExecutorRunRecorder(
            self,
            instance_id=instance_id,
            entry=entry,
            round=round_,
            emitter=emitter,
            observer=observer,
        )
        recorder.set_session_id(session_id)

        stream = executor.run(
            task or "", session_id=session_id, context=payload.get("context")
        )
        failure: Optional[BaseException] = None
        try:
            async for event in stream:
                if recorder.handle(event):
                    break
            if recorder.terminal_error is not None:
                failure = AgentError(recorder.terminal_error)
            return recorder.result()
        except AgentError as exc:
            failure = exc
            raise
        except Exception as exc:  # noqa: BLE001
            failure = AgentError(
                f"AgentExecutor {type(executor).__name__} raised "
                f"{type(exc).__name__}: {exc}"
            )
            raise failure from exc
        finally:
            try:
                if failure is not None:
                    recorder.fail(failure)
                recorder.flush()
            finally:
                observer.finish(failure)
                if emitter is not None:
                    emitter.close()
                await stream.aclose()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _record_executor_task(
        self, instance_id: str, task: Optional[str], source: Optional[str], entry: Any
    ) -> None:
        if not task:
            return
        user_message = {"role": "user", "content": task}
        self._process_user_message(
            instance_id, task, user_message, entry=entry, skip_save=True
        )
        if not self.orchestrator:
            self.text_formatter.print_message(
                self._label_message_with_source(user_message, source)
            )

    def _bind_executor(
        self, instance_id: str, round_: int, entry: Any
    ) -> AgentExecutorBase:
        """Return ``self.executor`` bound to this agent for one run."""
        tools = bridge_workflow_tools(
            self.tool_executor.list_tools(),
            ActivityToolContext(
                client_factory=self._get_wf_client,
                instance_id=instance_id,
                round=round_,
                source_agent=self.name,
                stream_context=getattr(entry, "stream_context", None),
            ),
        )
        hooks = self._hooks.before_tool_call if self._hooks else ()
        binding = ExecutorBinding(
            agent_name=self.name,
            system_prompt=self._executor_system_prompt(),
            max_iterations=self.execution.max_iterations,
            tools=tuple(tools),
            before_tool_call=tuple(hooks),
            session_store=self._executor_session_store(),
        )
        return self.executor.bind(binding)

    def _executor_system_prompt(self) -> Optional[str]:
        """Render the agent's system prompt (profile or explicit prompt)."""
        try:
            messages = self.prompting_helper.build_initial_messages(chat_history=[])
        except Exception:  # noqa: BLE001 - a prompt is optional for executors
            logger.exception("Could not render system prompt for %s", self.name)
            return None
        parts = [
            str(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        ]
        return "\n\n".join(parts) or None

    def _executor_session_store(self) -> Optional[DaprSessionStore]:
        """The agent's durable executor session store (on its state store)."""
        if not self.state_store:
            return None
        store = getattr(self, "_executor_store", None)
        if store is None:
            store = DaprSessionStore(
                self.state_store, config=DaprSessionStoreConfig(project_key=self.name)
            )
            self._executor_store = store
        return store

    def _executor_stream_emitter(self, instance_id: str, round_: int, entry: Any):
        stream_ctx = getattr(entry, "stream_context", None)
        if not stream_ctx:
            return None
        return self._build_stream_emitter(
            stream_ctx=stream_ctx,
            instance_id=instance_id,
            turn=round_ + 1,
            phase=None,
        )

    def _executor_observer(self, info: ExecutorRunInfo) -> ExecutorRunObserver:
        """Observer for one run; replaced by the OpenTelemetry instrumentor."""
        return ExecutorRunObserver()


def _executor_model(executor: AgentExecutorBase) -> Optional[str]:
    config = getattr(executor, "config", None)
    model = getattr(config, "model", None)
    return model if isinstance(model, str) else None


__all__ = ["DurableExecutorMixin"]
