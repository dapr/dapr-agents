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
Read-only helpers over Claude Code transcript entries.

A transcript is the ordered list of JSONL entries a ``SessionStore`` holds
for one session. These helpers inspect it before a run so the executor can
decide how to resume (a pending deferred tool call, a run that already
finished) and compute per-run cost deltas. They are pure functions over
plain dicts and do not import ``claude_agent_sdk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

_DEFERRED_ATTACHMENT = "hook_deferred_tool"


@dataclass(frozen=True)
class DeferredCall:
    """A tool call the transcript records as deferred and not yet resolved."""

    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


def _content_blocks(entry: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    message = entry.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, list):
        return ()
    return [block for block in content if isinstance(block, Mapping)]


def _tool_result_ids(entry: Mapping[str, Any]) -> Set[str]:
    return {
        str(block.get("tool_use_id"))
        for block in _content_blocks(entry)
        if block.get("type") == "tool_result"
    }


def _deferred_call(entry: Mapping[str, Any]) -> Optional[DeferredCall]:
    attachment = entry.get("attachment")
    if entry.get("type") != "attachment" or not isinstance(attachment, Mapping):
        return None
    if attachment.get("type") != _DEFERRED_ATTACHMENT:
        return None
    tool_input = attachment.get("toolInput")
    return DeferredCall(
        tool_call_id=str(attachment.get("toolUseID") or ""),
        name=str(attachment.get("toolName") or ""),
        arguments=dict(tool_input) if isinstance(tool_input, Mapping) else {},
    )


def pending_deferred_call(
    entries: Sequence[Mapping[str, Any]],
) -> Optional[DeferredCall]:
    """
    Return the most recent deferred tool call that has no result yet.

    Args:
        entries: Transcript entries in append order.

    Returns:
        The pending ``DeferredCall``, or ``None`` when the last deferral (if
        any) was already resolved by a later ``tool_result``.
    """
    pending: Optional[DeferredCall] = None
    for entry in entries:
        deferred = _deferred_call(entry)
        if deferred is not None and deferred.tool_call_id:
            pending = deferred
        elif (
            pending is not None
            and entry.get("type") == "user"
            and pending.tool_call_id in _tool_result_ids(entry)
        ):
            pending = None
    return pending


def final_assistant_text(entries: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """
    Return the final assistant text when the transcript ends on one.

    Walks back from the end over conversation entries. The CLI writes one
    assistant entry per content block, so consecutive text blocks of the
    same message are joined. Returns ``None`` when the run did not finish:
    the last conversation entry is a user entry, or the final assistant
    message also requested a tool call (its text is not the answer). Also
    returns ``None`` when the final message has no text.
    """
    texts: List[str] = []
    message_id: Optional[str] = None
    for entry in reversed(entries):
        kind = entry.get("type")
        if kind == "user":
            break
        if kind != "assistant":
            continue
        message = entry.get("message")
        entry_id = message.get("id") if isinstance(message, Mapping) else None
        if message_id is not None and entry_id != message_id:
            break
        message_id = entry_id
        blocks = _content_blocks(entry)
        if any(block.get("type") == "tool_use" for block in blocks):
            return None
        texts.extend(
            str(block.get("text", ""))
            for block in reversed(blocks)
            if block.get("type") == "text"
        )
    joined = "".join(reversed(texts)).strip()
    return joined or None


def last_total_cost(entries: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Return the session's cumulative cost from the last ``cost-state`` entry."""
    for entry in reversed(entries):
        if entry.get("type") == "cost-state":
            value = entry.get("totalCostUSD")
            if isinstance(value, (int, float)):
                return float(value)
            return None
    return None
