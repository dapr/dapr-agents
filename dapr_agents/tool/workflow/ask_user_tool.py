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

"""Built-in ``ask_user`` tool for mid-stream user input.

The tool suspends the workflow via ``wait_for_external_event`` and emits a
``USER_INPUT_REQUESTED`` chunk on the session stream so a UI or operator can
route the question to the user. When the user answers, the workflow resumes
and the tool returns the response string to the calling LLM as a ToolMessage.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Generator, Optional

from durabletask.task import TaskFailedError  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from dapr_agents.streaming.keys import USER_INPUT_EVENT_PREFIX
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool

logger = logging.getLogger(__name__)

ASK_USER_TOOL_NAME = "ask_user"
_ASK_USER_TIMEOUT_SENTINEL = "[user did not respond within timeout]"

#: Maximum length of a user-supplied answer, enforced both at the HTTP
#: ingress (``POST /input``) and in the tool's closure after wake. Prevents
#: the LLM context from being flooded by an adversarial responder.
ASK_USER_ANSWER_MAX_LEN = 4096


class AskUserArgs(BaseModel):
    """Arguments the LLM provides when invoking ``ask_user``."""

    question: str = Field(
        ...,
        description="The question to ask the user. Should be clear and specific.",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=1,
        description=(
            "How long to wait for the user's reply before giving up and "
            "returning a sentinel. Defaults to 10 minutes."
        ),
    )


def _make_ask_user_impl(publish_stream_event: Any):
    """Create the workflow-body closure for ``ask_user`` bound to an agent.

    ``publish_stream_event`` is the agent's activity callable (registered by
    name in ``register_workflows``). Binding it via closure avoids relying on
    hidden attributes of the Dapr workflow context and keeps the tool pure.
    """

    def _ask_user_impl(
        question: str,
        timeout_seconds: int = 600,
        ctx: Any = None,
        **_: Any,
    ) -> Generator[Any, Any, str]:
        if ctx is None:
            raise RuntimeError("ask_user requires the workflow ctx to be injected")

        new_uuid = getattr(ctx, "new_uuid", None)
        if callable(new_uuid):
            request_id = str(new_uuid())
        else:  # pragma: no cover - defensive fallback
            import uuid

            request_id = uuid.uuid4().hex

        instance_id = getattr(ctx, "instance_id", None)

        if instance_id is not None:
            yield ctx.call_activity(
                publish_stream_event,
                input={
                    "instance_id": instance_id,
                    "event_type": "user_input_requested",
                    "event_data": {
                        "request_id": request_id,
                        "question": question,
                        "target_instance_id": instance_id,
                        "timeout_seconds": timeout_seconds,
                    },
                },
            )

        event_name = f"{USER_INPUT_EVENT_PREFIX}:{request_id}"
        try:
            answer = yield ctx.wait_for_external_event(
                event_name, timeout=timedelta(seconds=timeout_seconds)
            )
        except TaskFailedError as exc:
            logger.info(
                "ask_user timed out (instance=%s request=%s): %s",
                instance_id,
                request_id,
                exc,
            )
            if instance_id is not None:
                yield ctx.call_activity(
                    publish_stream_event,
                    input={
                        "instance_id": instance_id,
                        "event_type": "user_input_timed_out",
                        "event_data": {"request_id": request_id},
                    },
                )
            return _ASK_USER_TIMEOUT_SENTINEL

        # Defence-in-depth: truncate answers that exceed the documented cap
        # even if the HTTP ingress missed it (raise_event can also arrive from
        # pub/sub or external callers outside the /input endpoint).
        if isinstance(answer, str) and len(answer) > ASK_USER_ANSWER_MAX_LEN:
            answer = answer[:ASK_USER_ANSWER_MAX_LEN]

        if instance_id is not None:
            yield ctx.call_activity(
                publish_stream_event,
                input={
                    "instance_id": instance_id,
                    "event_type": "user_input_received",
                    "event_data": {"request_id": request_id, "answer": answer},
                },
            )
        # Wrap the answer so injection prompts cannot start at byte 0 of the
        # ToolMessage fed back to the LLM. The wrapper is part of the
        # contract; downstream prompts should interpret the returned string
        # as an attributed user utterance, not an authoritative directive.
        return f"User responded: {answer}"

    return _ask_user_impl


def build_ask_user_tool(publish_stream_event: Any) -> WorkflowContextInjectedTool:
    """Construct the default ``ask_user`` tool instance for an agent.

    Args:
        publish_stream_event: The agent's ``publish_stream_event`` activity
            callable. Used to emit ``USER_INPUT_*`` chunks via the session
            listener.
    """

    return WorkflowContextInjectedTool(
        name=ASK_USER_TOOL_NAME,
        description=(
            "Ask the human user a direct question and wait for their "
            "response before continuing. Use this when you need information "
            "that only the user can provide (e.g., their preferences, "
            "the list of participants, a decision). Prefer concrete, "
            "answerable questions."
        ),
        args_model=AskUserArgs,
        func=_make_ask_user_impl(publish_stream_event),
    )


__all__ = [
    "ASK_USER_TOOL_NAME",
    "AskUserArgs",
    "build_ask_user_tool",
    "USER_INPUT_EVENT_PREFIX",
]
