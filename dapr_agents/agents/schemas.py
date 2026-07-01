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

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from dapr_agents.streaming.keys import StreamContextDict
from dapr_agents.types import MessageContent, ToolExecutionRecord
from dapr_agents.types.message import BaseMessage


def utcnow() -> datetime:
    """Return current time as timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ApprovalRequiredEvent(BaseModel):
    """
    Event published to Dapr Pub/Sub when a step requires human approval.

    The workflow publishes this event and suspends. An external approval service
    (Slack bot, web UI, CLI) receives it, notifies a human, and sends back an
    ApprovalResponseEvent to resume the workflow.

    The primary field is `step_name`. The `tool_name` property is kept for
    backward compatibility — existing listeners that read `event.tool_name`
    continue to work without changes.
    """

    approval_request_id: str = Field(
        description="Deterministic UUID for this request, stable across workflow replays"
    )
    instance_id: str = Field(
        description="Workflow instance that is waiting for approval"
    )
    step_name: str = Field(
        description="Name of the step that needs approval (tool name or 'llm')"
    )
    step_kind: str = Field(default="tool", description="'tool' or 'llm'")
    source: str = Field(
        default="local",
        description="Where the tool came from: 'local', 'mcp', 'openapi', etc.",
    )
    tool_call_id: str = Field(
        description="LLM-assigned ID of the tool call (from the assistant message)"
    )
    tool_arguments: Dict[str, Any] = Field(
        description="Arguments the LLM wants to pass to the tool"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Human-readable instructions shown to the approver",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the approval decision",
    )
    requested_at: datetime = Field(
        default_factory=utcnow, description="When approval was requested"
    )
    timeout_seconds: int = Field(
        description="Seconds before the workflow auto-denies if no human responds"
    )
    required_approver_scopes: List[str] = Field(
        default_factory=list,
        description=(
            "Scopes the approver's JWT must carry. Empty list = any verified "
            "approver. Approval UIs / approver-finding code can filter by these. "
            "Enforced by the downstream HITL plugin."
        ),
    )
    allowed_approver_subjects: List[str] = Field(
        default_factory=list,
        description=(
            "Optional allowlist of approver `sub` claim values. Empty list = "
            "any subject that meets required_approver_scopes."
        ),
    )
    approver_audience: Optional[str] = Field(
        default=None,
        description=(
            "Expected `aud` claim on the approver's JWT. None = falls back to "
            "the agent's own identity."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _compat_tool_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "tool_name" in data and "step_name" not in data:
            data = dict(data)
            data["step_name"] = data.pop("tool_name")
        return data

    @computed_field
    @property
    def tool_name(self) -> str:
        """backward compat alias for step_name — existing listeners can keep reading this."""
        return self.step_name


class ApprovalResponseEvent(BaseModel):
    """
    Event sent back to a waiting workflow with the human's approval decision.

    External approval services construct and send this event using
    DurableAgent.raise_approval_event() after a human reviews the request.
    """

    approval_request_id: str = Field(
        description="ID matching the original ApprovalRequiredEvent"
    )
    approved: bool = Field(description="True if approved, False if not approved")
    reason: Optional[str] = Field(
        default=None, description="Human-provided reason for the decision"
    )
    decided_at: datetime = Field(
        default_factory=utcnow, description="When the decision was made"
    )
    approver_token: Optional[str] = Field(
        default=None,
        repr=False,
        description=(
            "Sensitive: raw JWT submitted by the approver via the approval UI, CLI, "
            "or chat bot. dapr-agents carries it through the event payload as-is and "
            "does not validate or persist it. The downstream HITL plugin verifies it "
            "against the requirements declared on the original ApprovalRequiredEvent, "
            "and is responsible for scrubbing it so only the verified-claims subset "
            "(subject, scopes) propagates into any retained workflow history."
        ),
    )
    approver_subject: Optional[str] = Field(
        default=None,
        description=(
            "Approver's verified `sub` claim. Populated by the HITL plugin after "
            "verifying approver_token; not trusted on input from the raw event payload. "
            "Carrying it here lets observers see who approved without re-verifying."
        ),
    )


class BroadcastMessage(BaseMessage):
    """
    Represents a broadcast message from an agent.
    """


class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """

    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class TriggerAction(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """

    task: Optional[str] = Field(
        None,
        description="The specific task to execute. If not provided, the agent will act based on its memory or predefined behavior.",
    )
    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )
    caller_headers: Optional[Dict[str, str]] = Field(
        default=None,
        exclude=True,
        description=(
            "HTTP headers from the inbound request that triggered the agent. "
            "Used by lifecycle plugins on BEFORE_AGENT_INVOKE "
            "to extract the bearer token and other caller context. "
            "Populated by the agent's HTTP/serve layer when the trigger arrives via "
            "HTTP; left None for in-process or replay invocations. "
            "Transient and in-memory only: excluded from serialization (exclude=True) "
            "so raw headers such as Authorization are never emitted by model_dump(), "
            "never published over pub/sub, and never persisted to signed workflow "
            "history or logs. Plugins read it from the live object on "
            "BEFORE_AGENT_INVOKE."
        ),
    )


class AgentWorkflowMessage(MessageContent):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Timestamp when the message was created",
    )


class ConversationSummary(BaseModel):
    """
    Structured summary of a conversation for long-term memory storage.
    Used as response_format when generating summaries so the LLM returns
    parseable, consistent output that is easier to store and reuse.
    """

    summary: str = Field(
        ...,
        description="Concise summary of the conversation and tool usage for long-term memory: key facts, decisions, and outcomes.",
    )


class AgentWorkflowEntry(BaseModel):
    """
    Workflow entry data stored in the agent state store.
    Excludes fields provided by Dapr get_workflow.
    """

    messages: List[AgentWorkflowMessage] = Field(
        default_factory=list,
        description="Messages exchanged during the workflow (user, assistant, or tool messages).",
    )
    system_messages: List[AgentWorkflowMessage] = Field(
        default_factory=list,
        description="Rendered system prompt messages included when invoking the LLM.",
    )
    last_message: Optional[AgentWorkflowMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    triggering_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID of the entity that triggered this agent (for multi-agent communication).",
    )
    trace_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenTelemetry trace context for workflow resumption.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for resumable, multi-turn runs.",
    )
    approval_requests: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Tracks approval requests that have already been published, keyed by "
            "approval_request_id. Prevents duplicate publishes when the workflow replays."
        ),
    )
    pending_inputs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Outstanding ask_user requests awaiting a human answer, keyed by "
            "request_id. Persisted so a client reconnecting after a restart can "
            "recover the prompt (request_id + target_instance_id) via the status "
            "route or a /stream reattach. Cleared when the answer arrives or the "
            "request times out."
        ),
    )
    stream_context: Optional[StreamContextDict] = Field(
        default=None,
        description=(
            "Streaming context for this workflow instance — a structured "
            "``StreamContextDict`` (see ``dapr_agents.streaming.keys``), not a "
            "plain string: it carries ``root_instance_id``, the ``listener_config`` "
            "dict, parent agent metadata, ``depth``, ``call_path``, and trace "
            "correlation. Populated at session root and inherited by every "
            "descendant agent. ``None`` when streaming is not enabled for this run."
        ),
    )
