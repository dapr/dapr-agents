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
A ``DurableAgent`` conversation, derived from its workflow history.

The workflow history is the only durable copy of the conversation. Each
``call_llm`` activity receives just the messages added since the previous call
(``new_messages``), plus ``history_len`` and ``last_id``, which identify the
conversation before them. The activity keeps the conversation per instance in
memory and rebuilds it with ``fold_conversation`` from the sidecar's history
when its copy is missing or stale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Transcript:
    """
    Workflow-side cursor over the conversation.

    It lives in the workflow body, so replay rebuilds it from the same
    activity results every time.

    Attributes:
        length: Messages in the conversation before ``pending``.
        last_id: ``id`` of the last assistant message, ``None`` before the first.
        pending: Messages added since the previous ``call_llm``.
    """

    length: int = 0
    last_id: Optional[str] = None
    pending: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, *messages: Dict[str, Any]) -> None:
        self.pending.extend(messages)

    def payload(self) -> Dict[str, Any]:
        """The ``call_llm`` input keys describing the conversation."""
        return {
            "new_messages": list(self.pending),
            "history_len": self.length,
            "last_id": self.last_id,
        }

    def advance(self, assistant_message: Dict[str, Any]) -> None:
        """Account for a completed ``call_llm`` and its assistant message."""
        self.length += len(self.pending) + 1
        self.pending = []
        self.last_id = assistant_message.get("id")


def fold_conversation(
    events: Iterable[Any], call_llm_name: str
) -> List[Dict[str, Any]]:
    """
    Rebuild a conversation from workflow history events.

    Every completed ``call_llm`` activity adds its input's ``new_messages``
    and then its result. Failed or unfinished attempts add nothing, the same
    as the workflow's ``Transcript`` counts them.

    Args:
        events: History events, oldest first (``HistoryEvent`` protos).
        call_llm_name: Registered name of the agent's ``call_llm`` activity.

    Returns:
        The conversation, oldest message first.
    """
    inputs: Dict[int, str] = {}
    messages: List[Dict[str, Any]] = []
    for event in events:
        if event.HasField("taskScheduled"):
            if event.taskScheduled.name == call_llm_name:
                inputs[event.eventId] = event.taskScheduled.input.value
        elif event.HasField("taskCompleted"):
            raw = inputs.pop(event.taskCompleted.taskScheduledId, None)
            if raw is not None:
                payload = json.loads(raw) if raw else {}
                messages.extend(payload.get("new_messages") or [])
                messages.append(json.loads(event.taskCompleted.result.value))
    return messages


__all__ = ["Transcript", "fold_conversation"]
