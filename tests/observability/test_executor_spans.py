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

"""Tests for the OpenTelemetry spans emitted for executor runs."""

import pytest
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from dapr_agents.agents.executors import AgentEvent
from dapr_agents.agents.executors.observer import (
    ExecutorRunInfo,
    ExecutorRunObserver,
)
from dapr_agents.observability.wrappers.executor import (
    ExecutorObserverWrapper,
    TracingExecutorObserver,
)

INFO = ExecutorRunInfo(
    agent_name="assistant",
    executor_type="ClaudeAgentExecutor",
    instance_id="wf-1",
    round=1,
    session_id="s0",
    model="claude-x",
)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _spans(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def _tool_call(call_id, name="get_weather"):
    return AgentEvent(
        type="tool_call",
        content={"id": call_id, "name": name, "arguments": {"city": "P"}},
        session_id="s1",
    )


def _tool_result(call_id, is_error=False):
    return AgentEvent(
        type="tool_result",
        content={"tool_call_id": call_id, "result": "sunny", "is_error": is_error},
    )


class TestTracingObserver:
    def test_agent_span_with_tool_children_and_usage(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.on_event(_tool_call("t1"))
        observer.on_event(_tool_result("t1"))
        observer.on_event(
            AgentEvent(
                type="complete",
                content={"role": "assistant", "content": "done"},
                metadata={
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 7,
                    },
                    "cost_usd": 0.1,
                    "num_turns": 2,
                    "stop_reason": "end_turn",
                    "model_usage": {"claude-x-2026": {}},
                },
            )
        )
        observer.finish(None)

        spans = _spans(exporter)
        assert set(spans) == {"invoke_agent assistant", "execute_tool get_weather"}
        agent = spans["invoke_agent assistant"]
        attrs = agent.attributes
        assert attrs["gen_ai.agent.name"] == "assistant"
        assert attrs["gen_ai.request.model"] == "claude-x"
        assert attrs["gen_ai.response.model"] == "claude-x-2026"
        assert attrs["gen_ai.usage.input_tokens"] == 5
        assert attrs["llm.token_count.prompt"] == 5
        assert attrs["gen_ai.usage.output_tokens"] == 2
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 7
        assert attrs["executor.cost_usd"] == 0.1
        assert attrs["executor.num_turns"] == 2
        assert attrs["executor.round"] == 1
        assert attrs["session.id"] == "s1"
        assert agent.status.status_code is StatusCode.OK

        tool = spans["execute_tool get_weather"]
        assert tool.parent.span_id == agent.context.span_id
        assert tool.attributes["gen_ai.tool.call.id"] == "t1"
        assert tool.status.status_code is StatusCode.OK

    def test_failed_tool_result_marks_tool_span(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.on_event(_tool_call("t1"))
        observer.on_event(_tool_result("t1", is_error=True))
        observer.finish(None)
        tool = _spans(exporter)["execute_tool get_weather"]
        assert tool.status.status_code is StatusCode.ERROR
        assert tool.attributes["error.type"] == "tool_error"

    def test_paused_run_leaves_tool_open_then_closes_it(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.on_event(_tool_call("t1", name="transfer"))
        observer.on_event(
            AgentEvent(
                type="paused",
                content={"tool_call_id": "t1", "name": "transfer"},
                metadata={"cost_usd": 0.02},
            )
        )
        observer.finish(None)
        spans = _spans(exporter)
        agent = spans["invoke_agent assistant"]
        assert agent.attributes["executor.paused_tool_call_id"] == "t1"
        assert agent.attributes["executor.cost_usd"] == 0.02
        assert spans["execute_tool transfer"].attributes["tool.completed"] is False

    def test_error_event_sets_error_status(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.on_event(AgentEvent(type="error", content="boom"))
        observer.finish(None)
        agent = _spans(exporter)["invoke_agent assistant"]
        assert agent.status.status_code is StatusCode.ERROR
        assert agent.attributes["error.type"] == "executor_error"

    def test_exception_sets_error_type(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.finish(RuntimeError("crash"))
        agent = _spans(exporter)["invoke_agent assistant"]
        assert agent.status.status_code is StatusCode.ERROR
        assert agent.attributes["error.type"] == "RuntimeError"
        assert agent.events[0].name == "exception"

    def test_events_without_ids_are_ignored(self, tracer, exporter):
        observer = TracingExecutorObserver(tracer, INFO)
        observer.on_event(AgentEvent(type="tool_call", content={"name": "x"}))
        observer.on_event(AgentEvent(type="tool_result", content={"result": "r"}))
        observer.on_event(AgentEvent(type="text_delta", content="hi"))
        observer.finish(None)
        assert set(_spans(exporter)) == {"invoke_agent assistant"}


class TestObserverWrapper:
    def test_returns_tracing_observer(self, tracer):
        wrapper = ExecutorObserverWrapper(tracer)
        observer = wrapper(lambda info: ExecutorRunObserver(), None, (INFO,), {})
        assert isinstance(observer, TracingExecutorObserver)
        observer.finish(None)

    def test_accepts_keyword_info(self, tracer):
        wrapper = ExecutorObserverWrapper(tracer)
        observer = wrapper(lambda info: None, None, (), {"info": INFO})
        assert isinstance(observer, TracingExecutorObserver)
        observer.finish(None)

    def test_falls_back_without_tracer_or_info(self, tracer):
        default = ExecutorRunObserver()
        assert (
            ExecutorObserverWrapper(None)(lambda i: default, None, (INFO,), {})
            is default
        )
        assert (
            ExecutorObserverWrapper(tracer)(lambda i: default, None, ("x",), {})
            is default
        )

    def test_respects_suppressed_instrumentation(self, tracer):
        default = ExecutorRunObserver()
        token = otel_context.attach(
            otel_context.set_value(otel_context._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            observer = ExecutorObserverWrapper(tracer)(
                lambda i: default, None, (INFO,), {}
            )
        finally:
            otel_context.detach(token)
        assert observer is default


def test_default_observer_is_noop():
    observer = ExecutorRunObserver()
    observer.on_event(AgentEvent(type="text_delta", content="x"))
    observer.finish(RuntimeError("ignored"))
