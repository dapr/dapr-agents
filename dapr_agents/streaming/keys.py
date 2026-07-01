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

"""Shared streaming-protocol constants and the typed ``StreamContextDict``.

These symbols are cross-module glue: workflow body, activities, the runner,
and the HTTP surface all read/write the same metadata keys, and a single
typo would silently short-circuit streaming (every read uses ``.get()`` with
a default). Keeping the strings here is the single source of truth.

``StreamContextDict`` is the on-the-wire schema for ``entry.stream_context``
and the ``_stream_context`` metadata entry. It serialises as a plain JSON
object across the Dapr workflow boundary; ``TypedDict`` gives static type
checking at call sites without adding runtime cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# Metadata keys (inbound payload / activity input plumbing)
# ---------------------------------------------------------------------------

#: Outer envelope key on ``TriggerAction`` / workflow input payloads.
MESSAGE_METADATA: str = "_message_metadata"

#: Nested key inside ``_message_metadata`` carrying a full ``StreamContextDict``
#: inherited from a parent workflow (multi-agent descent).
STREAM_CONTEXT: str = "_stream_context"

#: Nested key inside ``_message_metadata`` stamped by the runner at root
#: scheduling time. Frozen alongside ``_include_complete_message`` so the
#: workflow body never reads mutable agent config during replay.
STREAM_LISTENER_CONFIG: str = "_stream_listener_config"

#: Nested key inside ``_message_metadata`` toggling whether terminal chunks
#: (``TURN_COMPLETE`` / ``SESSION_COMPLETE``) carry the full assistant message.
INCLUDE_COMPLETE_MESSAGE: str = "_include_complete_message"

#: Activity-payload key carrying an orchestrator's phase tag (``planning`` /
#: ``routing`` / ``evaluating`` / ``summarizing``). Passed to ``call_llm`` so
#: the emitter can attribute chunks to the phase.
STREAM_PHASE: str = "_stream_phase"

#: Prefix used by ``ask_user`` for the workflow event name raised via
#: ``DaprClient.raise_workflow_event``: ``user_input_response:{request_id}``.
USER_INPUT_EVENT_PREFIX: str = "user_input_response"


# ---------------------------------------------------------------------------
# StreamContextDict — typed schema for the stream_context dict
# ---------------------------------------------------------------------------


class StreamContextDict(TypedDict, total=False):
    """Typed shape for ``entry.stream_context`` and inherited ``_stream_context``.

    Fields are populated at session root by the runner (or by
    :meth:`DurableAgent._resolve_stream_context` when the inbound metadata
    already carries a parent-stamped context).

    All fields are ``total=False`` because older entries persisted before
    streaming was enabled (and test fixtures) may omit them; readers must
    use ``.get(..., default)`` rather than subscript access.
    """

    #: Workflow instance id of the user-facing entry agent. Stable for the
    #: lifetime of the session; used as ordering key on the stream topic.
    root_instance_id: str

    #: JSON-serialisable listener config dict consumed by
    #: ``dapr_agents.streaming.listeners.build_listener``. Shape depends on
    #: ``type``: see that factory for per-type schemas.
    listener_config: Dict[str, Any]

    #: Name of the immediate parent agent (``None`` at root). Children stamp
    #: this to ``self.name`` before dispatching agent-as-tool / sub-agent.
    parent_agent: Optional[str]

    #: Workflow instance id of the immediate parent (``None`` at root).
    parent_instance_id: Optional[str]

    #: Descent depth. ``0`` at root; incremented on each child dispatch.
    depth: int

    #: Ordered list of agent names from root to current. Children append
    #: their own name; never empty after ``_resolve_stream_context``.
    call_path: List[str]

    #: W3C ``traceparent`` header if OTel trace context was available. Lets
    #: the emitter stitch stream chunks to the span hierarchy.
    trace_parent: Optional[str]

    #: If ``True``, ``TURN_COMPLETE`` / ``SESSION_COMPLETE`` chunks carry the
    #: full ``AssistantMessage`` in ``complete_message``. Default ``False``
    #: so clients reconstruct from deltas.
    include_complete_message: bool


__all__ = [
    "INCLUDE_COMPLETE_MESSAGE",
    "MESSAGE_METADATA",
    "STREAM_CONTEXT",
    "STREAM_LISTENER_CONFIG",
    "STREAM_PHASE",
    "StreamContextDict",
    "USER_INPUT_EVENT_PREFIX",
]
