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
Spans for ``AgentExecutorBase`` runs driven by ``DurableAgent``.

``ExecutorObserverWrapper`` wraps ``DurableExecutorMixin._executor_observer``
so every ``run_executor`` activity gets a ``TracingExecutorObserver``. The
observer emits one ``invoke_agent {agent}`` AGENT span per run (a child of
the activity span) and one ``execute_tool {tool}`` TOOL span per tool call
the executor reports, dual-emitting OpenInference and GenAI semconv
attributes like the other wrappers. Token usage and cost from the terminal
event are recorded on the agent span.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dapr_agents.agents.executors import event as ev
from dapr_agents.agents.executors.event import AgentEvent
from dapr_agents.agents.executors.observer import ExecutorRunInfo, ExecutorRunObserver

from ..constants import (
    AGENT,
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    TOOL,
    GenAiOperationNameValues,
    Status,
    StatusCode,
    context_api,
    safe_json_dumps,
    trace_api,
)

SESSION_ID = "session.id"
_EXECUTOR_ERROR = "executor_error"
_TOOL_ERROR = "tool_error"

# ResultMessage-style usage keys -> span attributes (GenAI + OpenInference).
_USAGE_ATTRIBUTES = {
    "input_tokens": (GEN_AI_USAGE_INPUT_TOKENS, LLM_TOKEN_COUNT_PROMPT),
    "output_tokens": (GEN_AI_USAGE_OUTPUT_TOKENS, LLM_TOKEN_COUNT_COMPLETION),
    "cache_creation_input_tokens": (GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,),
    "cache_read_input_tokens": (GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,),
}
_RUN_ATTRIBUTES = {
    "cost_usd": "executor.cost_usd",
    "session_total_cost_usd": "executor.session_total_cost_usd",
    "num_turns": "executor.num_turns",
    "stop_reason": "executor.stop_reason",
}


def _usage_attributes(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {}
    usage = metadata.get("usage")
    if isinstance(usage, Mapping):
        for key, names in _USAGE_ATTRIBUTES.items():
            value = usage.get(key)
            if isinstance(value, int):
                attributes.update({name: value for name in names})
    for key, name in _RUN_ATTRIBUTES.items():
        value = metadata.get(key)
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            attributes[name] = value
    model_usage = metadata.get("model_usage")
    if isinstance(model_usage, Mapping) and model_usage:
        attributes[GEN_AI_RESPONSE_MODEL] = str(next(iter(model_usage)))
    return attributes


class TracingExecutorObserver(ExecutorRunObserver):
    """Emits the agent and tool spans for one executor run."""

    def __init__(self, tracer: Any, info: ExecutorRunInfo) -> None:
        self._tracer = tracer
        attributes: Dict[str, Any] = {
            OPENINFERENCE_SPAN_KIND: AGENT,
            GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT,
            GEN_AI_AGENT_NAME: info.agent_name,
            "agent.name": info.agent_name,
            "executor.type": info.executor_type,
            "executor.round": info.round,
            "workflow.instance_id": info.instance_id,
        }
        if info.session_id:
            attributes[SESSION_ID] = info.session_id
        if info.model:
            attributes[GEN_AI_REQUEST_MODEL] = info.model
        self._span = tracer.start_span(
            f"invoke_agent {info.agent_name}", attributes=attributes
        )
        self._span_context = trace_api.set_span_in_context(self._span)
        self._tool_spans: Dict[str, Any] = {}
        self._error: Optional[str] = None

    def on_event(self, event: AgentEvent) -> None:
        if event.session_id:
            self._span.set_attribute(SESSION_ID, event.session_id)
        content = event.content if isinstance(event.content, Mapping) else {}
        match event.type:
            case ev.EVENT_TOOL_CALL:
                self._start_tool(content)
            case ev.EVENT_TOOL_RESULT:
                self._end_tool(content)
            case ev.EVENT_COMPLETE:
                self._span.set_attribute(
                    OUTPUT_VALUE, safe_json_dumps(content.get("content", ""))
                )
                self._span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                self._span.set_attributes(_usage_attributes(event.metadata))
            case ev.EVENT_PAUSED:
                self._span.set_attribute(
                    "executor.paused_tool_call_id", str(content.get("tool_call_id"))
                )
                self._span.set_attributes(_usage_attributes(event.metadata))
            case ev.EVENT_ERROR:
                self._error = str(event.content)

    def _start_tool(self, call: Mapping[str, Any]) -> None:
        call_id = str(call.get("id") or "")
        if not call_id:
            return
        name = str(call.get("name") or "unknown_tool")
        self._tool_spans[call_id] = self._tracer.start_span(
            f"execute_tool {name}",
            context=self._span_context,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL,
                GEN_AI_TOOL_NAME: name,
                GEN_AI_TOOL_CALL_ID: call_id,
                "tool.name": name,
                INPUT_VALUE: safe_json_dumps(call.get("arguments") or {}),
                INPUT_MIME_TYPE: "application/json",
            },
        )

    def _end_tool(self, result: Mapping[str, Any]) -> None:
        call_id = str(result.get("tool_call_id") or result.get("tool_use_id") or "")
        if call_id and call_id not in self._tool_spans:
            # A resumed run reports the approved call's result without its
            # tool_use block, which was emitted by the run that paused.
            self._start_tool({"id": call_id, "name": result.get("name")})
            self._tool_spans[call_id].set_attribute("tool.resumed", True)
        span = self._tool_spans.pop(call_id, None)
        if span is None:
            return
        span.set_attribute(OUTPUT_VALUE, safe_json_dumps(result.get("result")))
        span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
        if result.get("is_error"):
            span.set_status(Status(StatusCode.ERROR, str(result.get("result"))))
            span.set_attribute("error.type", _TOOL_ERROR)
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    def finish(self, error: Optional[BaseException] = None) -> None:
        # Calls still open were deferred for approval or cut off by a failure.
        for span in self._tool_spans.values():
            span.set_attribute("tool.completed", False)
            span.end()
        self._tool_spans.clear()
        if error is not None:
            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            self._span.set_attribute("error.type", type(error).__qualname__)
            self._span.record_exception(error)
        elif self._error is not None:
            self._span.set_status(Status(StatusCode.ERROR, self._error))
            self._span.set_attribute("error.type", _EXECUTOR_ERROR)
        else:
            self._span.set_status(Status(StatusCode.OK))
        self._span.end()


class ExecutorObserverWrapper:
    """Wraps ``DurableExecutorMixin._executor_observer`` to return a tracer."""

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def __call__(self, wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
        if context_api and context_api.get_value(
            context_api._SUPPRESS_INSTRUMENTATION_KEY
        ):
            return wrapped(*args, **kwargs)
        info = args[0] if args else kwargs.get("info")
        if not isinstance(info, ExecutorRunInfo) or self._tracer is None:
            return wrapped(*args, **kwargs)
        return TracingExecutorObserver(self._tracer, info)


__all__ = ["ExecutorObserverWrapper", "TracingExecutorObserver"]
