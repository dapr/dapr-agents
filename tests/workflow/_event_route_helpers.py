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

"""Shared helpers for the workflow event route tests."""

from __future__ import annotations

from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import grpc
from dapr.ext.workflow.workflow_state import WorkflowStatus
from pydantic import BaseModel

from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.types.workflow import (
    DEFAULT_EVENT_CALL_TIMEOUT_SECONDS,
    NotFoundRetryPolicy,
    WorkflowEventRouteSpec,
)
from dapr_agents.workflow.utils.event_routes import (
    EventRouteTarget,
    WorkflowEventDispatcher,
)
from dapr_agents.workflow.utils.subscription import MessageContext

PATCH_TARGET = "dapr_agents.workflow.utils.registration.default_dapr_client_factory"


class FakeRpcError(grpc.RpcError):
    """A gRPC error with a fixed status code and details."""

    def __init__(self, code: Any, details: str = "") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> Any:
        return self._code

    def details(self) -> str:
        return self._details


class BrokenRpcError(grpc.RpcError):
    """A gRPC error whose accessors raise."""

    def code(self) -> Any:
        raise RuntimeError("boom")

    def details(self) -> str:
        raise RuntimeError("boom")


class JobRef(BaseModel):
    workflow_id: str


class JobFinished(BaseModel):
    job: JobRef
    status: str


def workflow_state(status: Any) -> MagicMock:
    """A WorkflowState stand-in with the given runtime status."""
    state = MagicMock()
    state.runtime_status = status
    return state


def make_spec(defaults: dict[str, Any], **overrides: Any) -> WorkflowEventRouteSpec:
    """Build a spec on pub/sub ``messagepubsub`` / topic ``t`` from defaults + overrides."""
    values: dict[str, Any] = {"pubsub_name": "messagepubsub", "topic": "t"}
    values.update(defaults)
    values.update(overrides)
    return WorkflowEventRouteSpec(**values)


def make_ctx(event_id: Optional[str] = "evt-1", name: str = "route") -> MessageContext:
    """A MessageContext on topic ``t`` with the given CloudEvent id."""
    fields = dict.fromkeys(EventMessageMetadata.model_fields)
    fields.update(id=event_id, topic="t")
    return MessageContext(
        event=EventMessageMetadata.model_validate(fields), handler_name=name
    )


def make_target(
    *,
    call_timeout_seconds: float = DEFAULT_EVENT_CALL_TIMEOUT_SECONDS,
    **overrides: Any,
) -> EventRouteTarget:
    """An event target raising ``evt`` on the instance at field ``wf_id``."""
    values: dict[str, Any] = dict(
        event_name="evt",
        instance_id_from="wf_id",
        event_name_from=None,
        data_from=None,
        dedupe=True,
        deduper=None,
        not_found_retry=NotFoundRetryPolicy(),
        call_timeout_seconds=call_timeout_seconds,
    )
    values.update(overrides)
    return EventRouteTarget(**values)


def make_wf(status: Optional[WorkflowStatus] = WorkflowStatus.RUNNING) -> MagicMock:
    """A workflow client whose instance has ``status`` (None: not found)."""
    wf_client = MagicMock()
    wf_client.get_workflow_state.return_value = (
        workflow_state(status) if status is not None else None
    )
    return wf_client


def make_dispatcher(
    wf_client: Any,
    *,
    serializer: Callable[[Any], Any] = lambda message: message,
    **kwargs: Any,
) -> WorkflowEventDispatcher:
    """A dispatcher over ``wf_client``; ``kwargs`` go to the constructor."""
    return WorkflowEventDispatcher(
        wf_client=wf_client, default_serializer=serializer, **kwargs
    )


def run_dispatch(
    dispatcher: WorkflowEventDispatcher,
    *,
    target: Optional[EventRouteTarget] = None,
    message: Any = None,
    ctx: Optional[MessageContext] = None,
    dlq: Optional[str] = None,
    dedupe_key: Optional[str] = None,
    deduper: Any = None,
) -> str:
    """Dispatch on ``messagepubsub`` / ``t``; the message defaults to ``wf-1``.

    ``dedupe_key`` is the message key (CloudEvent id or payload digest).
    """
    return dispatcher.dispatch(
        target=target or make_target(),
        route_name="route",
        pubsub="messagepubsub",
        topic="t",
        dead_letter_topic=dlq,
        message=message if message is not None else {"wf_id": "wf-1"},
        msg_ctx=ctx or make_ctx(),
        message_key=dedupe_key,
        deduper=deduper,
    )


def only_tracked_raise(dispatcher: WorkflowEventDispatcher) -> Any:
    """The single timed-out raise tracked on ``messagepubsub`` / ``t``."""
    timed_out = dispatcher._topic_states[("messagepubsub", "t")].timed_out
    tracked = [*timed_out._running.values(), *timed_out._settled.values()]
    assert len(tracked) == 1, tracked
    return tracked[0]
