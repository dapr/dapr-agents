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
Per-run bookkeeping for ``DurableAgent``'s ``run_executor`` activity.

``ExecutorRunRecorder`` applies the events of one executor run to the
workflow entry (messages, tool history, usage, checkpoints), forwards text
deltas to the session stream and reports every event to the run's
observer. ``PausedExecutorRun`` is the activity result for a run that
paused on a tool call awaiting approval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from dapr_agents.agents.executors import event as ev
from dapr_agents.agents.executors.event import AgentEvent
from dapr_agents.agents.executors.observer import ExecutorRunObserver
from dapr_agents.hooks import RequireApproval
from dapr_agents.streaming.emitter import StreamEmitter
from dapr_agents.types import AgentError
from dapr_agents.types.streaming import StreamChunkType
from dapr_agents.types.tools import ToolExecutionRecord, ToolExecutionStatus

logger = logging.getLogger(__name__)

# Key marking a ``run_executor`` activity result as a paused run. Activity
# results must be JSON, and a completed run returns the plain final message,
# so a paused run is a dict carrying only this key.
EXECUTOR_PAUSED_KEY = "_executor_paused"
EXECUTOR_FAILED_KEY = "_executor_failed"


def executor_failure(result: Any) -> Optional[str]:
    """The terminal error in a ``run_executor`` result, if it is one."""
    if isinstance(result, Mapping) and EXECUTOR_FAILED_KEY in result:
        return str(result[EXECUTOR_FAILED_KEY])
    return None


_USAGE_FIELDS = (
    "cost_usd",
    "session_total_cost_usd",
    "usage",
    "model_usage",
    "num_turns",
    "stop_reason",
)


@dataclass(frozen=True)
class PausedExecutorRun:
    """
    A run that ended with a ``paused`` event (a tool call awaits approval).

    Attributes:
        tool_call_id: Id of the deferred call; the decision must use it.
        name: Tool name as the agent's hooks see it.
        arguments: Tool arguments.
        session_id: Session to resume.
        approval: ``RequireApproval`` details (``timeout_seconds``,
            ``instructions``, ``reason``); empty when the executor paused
            without them.
        source: Where the tool comes from (``local``, ``mcp``, ...), when
            the executor reported it.
    """

    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    approval: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    @classmethod
    def from_event(cls, event: AgentEvent) -> "PausedExecutorRun":
        content = event.content if isinstance(event.content, Mapping) else {}
        tool_call_id = str(content.get("tool_call_id") or "")
        if not tool_call_id:
            raise AgentError("Executor paused without a tool_call_id.")
        return cls(
            tool_call_id=tool_call_id,
            name=str(content.get("name") or ""),
            arguments=dict(content.get("arguments") or {}),
            session_id=event.session_id,
            approval=dict(content.get("approval") or {}),
            source=_optional_str(content.get("source")),
        )

    @classmethod
    def from_activity_result(cls, result: Any) -> Optional["PausedExecutorRun"]:
        """Return the paused run in a ``run_executor`` result, if it is one."""
        if not isinstance(result, Mapping):
            return None
        data = result.get(EXECUTOR_PAUSED_KEY)
        if not isinstance(data, Mapping):
            return None
        if not data.get("tool_call_id"):
            raise AgentError("Paused executor result has no tool_call_id.")
        return cls(
            tool_call_id=str(data["tool_call_id"]),
            name=str(data.get("name") or ""),
            arguments=dict(data.get("arguments") or {}),
            session_id=data.get("session_id"),
            approval=dict(data.get("approval") or {}),
            source=_optional_str(data.get("source")),
        )

    def to_activity_result(self) -> Dict[str, Any]:
        return {
            EXECUTOR_PAUSED_KEY: {
                "tool_call_id": self.tool_call_id,
                "name": self.name,
                "arguments": dict(self.arguments),
                "session_id": self.session_id,
                "approval": dict(self.approval),
                "source": self.source,
            }
        }

    def tool_call(self) -> Dict[str, Any]:
        """The call in the OpenAI tool-call shape ``_request_approval`` takes."""
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, default=str),
            },
        }

    def require_approval(self) -> RequireApproval:
        return RequireApproval(
            timeout_seconds=self.approval.get("timeout_seconds"),
            instructions=self.approval.get("instructions"),
            reason=self.approval.get("reason"),
        )


def _optional_str(value: Any) -> Optional[str]:
    return str(value) if value else None


def _index_records(tool_history: Any) -> Dict[str, ToolExecutionRecord]:
    return {
        record.tool_call_id: record
        for record in tool_history or ()
        if record.tool_call_id
    }


def _as_message(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    return {"role": "assistant", "content": str(content)}


def _result_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


class ExecutorRunRecorder:
    """
    Applies one executor run's events to the workflow entry.

    Args:
        agent: The hosting ``DurableAgent``.
        instance_id: Workflow instance of the run.
        entry: The loaded workflow entry (refreshed on ``session`` events).
        round: Run round within the instance.
        emitter: Session stream emitter, or ``None`` when not streaming.
        observer: Receives every event.
    """

    def __init__(
        self,
        agent: Any,
        *,
        instance_id: str,
        entry: Any,
        round: int,
        emitter: Optional[StreamEmitter],
        observer: ExecutorRunObserver,
    ) -> None:
        self._agent = agent
        self._instance_id = instance_id
        self.entry = entry
        self._round = round
        self._emitter = emitter
        self._observer = observer
        # Seeded from the entry so a resumed run updates the record of the
        # call it paused on instead of appending a duplicate.
        self._tool_records = _index_records(getattr(entry, "tool_history", None))
        self.session_id: Optional[str] = getattr(entry, "session_id", None)
        self.final_message: Optional[Dict[str, Any]] = None
        self.paused: Optional[PausedExecutorRun] = None
        self.terminal_error: Optional[str] = None
        self._saved_texts: List[Any] = []

    # ------------------------------------------------------------------
    def set_session_id(self, session_id: Optional[str]) -> None:
        if not session_id or session_id == self.session_id:
            return
        self.session_id = session_id
        if hasattr(self.entry, "session_id"):
            self.entry.session_id = session_id

    def handle(self, event: AgentEvent) -> bool:
        """Apply one event; return ``True`` once the run reached its end.

        Raises:
            AgentError: On an ``error`` event.
        """
        self._observer.on_event(event)
        self.set_session_id(event.session_id)
        match event.type:
            case ev.EVENT_TEXT_DELTA:
                if self._emitter is not None and isinstance(event.content, str):
                    self._emitter.emit_text_delta(event.content)
            case ev.EVENT_MESSAGE:
                self._on_message(_as_message(event.content))
            case ev.EVENT_TOOL_CALL:
                self._on_tool_call(event.content)
            case ev.EVENT_TOOL_RESULT:
                self._on_tool_result(event.content)
            case ev.EVENT_SESSION:
                self._checkpoint()
            case ev.EVENT_COMPLETE:
                self._on_complete(event)
            case ev.EVENT_PAUSED:
                self._on_paused(event)
            case ev.EVENT_ERROR:
                message = f"AgentExecutor emitted error: {event.content}"
                if event.metadata.get(ev.METADATA_RETRYABLE) is not False:
                    raise AgentError(message)
                self.terminal_error = message
        return event.type in ev.TERMINAL_EVENT_TYPES

    # ------------------------------------------------------------------
    def _on_message(self, message: Dict[str, Any]) -> None:
        if message.get("role") != "assistant":
            return
        self._agent._save_assistant_message(
            self._instance_id, dict(message), entry=self.entry, skip_save=True
        )
        self._saved_texts.append(message.get("content"))
        if not self._agent.orchestrator:
            self._agent.text_formatter.print_message(message)

    def _on_tool_call(self, content: Any) -> None:
        call = content if isinstance(content, dict) else {}
        tc_id = str(call.get("id") or "")
        if not tc_id:
            logger.warning(
                "Executor emitted tool_call without an 'id'; skipping record "
                "(tool_name=%r).",
                call.get("name"),
            )
            return
        record = self._new_record(
            tc_id,
            str(call.get("name", "")),
            dict(call.get("arguments", {}) or {}),
            ToolExecutionStatus.RUNNING,
        )
        self._tool_records[tc_id] = record
        self.entry.tool_history.append(record)

    def _on_tool_result(self, content: Any) -> None:
        result = content if isinstance(content, dict) else {}
        # OpenAI-style (tool_call_id) and Anthropic-style (tool_use_id) ids.
        tc_id = str(result.get("tool_call_id") or result.get("tool_use_id") or "")
        if not tc_id:
            logger.warning(
                "Executor emitted tool_result without a tool_call_id/tool_use_id; "
                "skipping (tool_name=%r).",
                result.get("name"),
            )
            return
        status = (
            ToolExecutionStatus.FAILED
            if result.get("is_error")
            else ToolExecutionStatus.COMPLETED
        )
        record = self._tool_records.get(tc_id)
        if record is None:
            record = self._new_record(tc_id, str(result.get("name", "")), {}, status)
            self.entry.tool_history.append(record)
        record.status = status
        record.completed_at = datetime.now(timezone.utc)
        record.execution_result = _result_text(result.get("result"))

    def _new_record(
        self,
        tc_id: str,
        name: str,
        arguments: Dict[str, Any],
        status: ToolExecutionStatus,
    ) -> ToolExecutionRecord:
        return ToolExecutionRecord(
            tool_call_id=tc_id,
            tool_name=name,
            tool_args=arguments,
            status=status,
            is_agent_call=False,
            executing_agent=self._agent.name,
            agent_workflow_instance_id=self._instance_id,
        )

    def _checkpoint(self) -> None:
        """Persist what has accumulated and continue on the refreshed entry."""
        self._agent.save_state(self._instance_id, entry=self.entry)
        # get_state validates into new objects, so rebuild the record index
        # to keep later tool_result events updating persisted records.
        self.entry = self._agent._infra.get_state(self._instance_id)
        self._tool_records = _index_records(self.entry.tool_history)

    def _record_usage(self, event: AgentEvent) -> Dict[str, Any]:
        usage = {k: event.metadata[k] for k in _USAGE_FIELDS if k in event.metadata}
        if usage and hasattr(self.entry, "executor_usage"):
            self.entry.executor_usage.append(
                {"round": self._round, "session_id": self.session_id, **usage}
            )
        return usage

    def _on_complete(self, event: AgentEvent) -> None:
        self.final_message = _as_message(event.content)
        usage = self._record_usage(event)
        if self._emitter is not None:
            self._emitter.complete_turn(self.final_message, metadata=usage)

    def _on_paused(self, event: AgentEvent) -> None:
        paused = PausedExecutorRun.from_event(event)
        if paused.session_id is None:
            paused = replace(paused, session_id=self.session_id)
        self.paused = paused
        record = self._tool_records.get(paused.tool_call_id)
        if record is not None:
            record.status = ToolExecutionStatus.PENDING
        self._record_usage(event)
        if self._emitter is not None:
            self._emitter.emit_event(
                StreamChunkType.TURN_PAUSED,
                event_data={
                    "reason": "tool_approval",
                    "tool_call_id": paused.tool_call_id,
                    "tool_name": paused.name,
                },
            )

    # ------------------------------------------------------------------
    def fail(self, exc: BaseException) -> None:
        """Report a failed run on the stream."""
        if self._emitter is not None:
            self._emitter.emit_error(type(exc).__name__, str(exc))

    def flush(self) -> None:
        """Persist the entry, adding the final message if not yet recorded."""
        final = self.final_message
        if final is not None:
            if final.get("content") not in self._saved_texts:
                self._agent._save_assistant_message(
                    self._instance_id, dict(final), entry=self.entry, skip_save=True
                )
        self._agent.save_state(self._instance_id, entry=self.entry)

    def result(self) -> Dict[str, Any]:
        """The ``run_executor`` activity result.

        Raises:
            AgentError: If the run ended without a terminal event.
        """
        if self.terminal_error is not None:
            return {EXECUTOR_FAILED_KEY: self.terminal_error}
        if self.paused is not None:
            return self.paused.to_activity_result()
        if self.final_message is None:
            raise AgentError("AgentExecutor stream ended without a 'complete' event.")
        return self.final_message


__all__ = [
    "EXECUTOR_FAILED_KEY",
    "EXECUTOR_PAUSED_KEY",
    "executor_failure",
    "ExecutorRunRecorder",
    "PausedExecutorRun",
]
