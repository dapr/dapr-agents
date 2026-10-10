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

import inspect
from unittest.mock import patch

import pytest
import wrapt
from opentelemetry import trace

from dapr_agents.observability.instrumentor import DaprAgentsInstrumentor

# wrapt 1.x names the first parameter ``module``; wrapt 2.x renamed it to
# ``target``. Passing it positionally is the only form both accept.
WRAPT_SIGNATURE = inspect.signature(wrapt.wrap_function_wrapper)


@pytest.fixture
def recorded_calls():
    calls = []

    def spy(*args, **kwargs):
        WRAPT_SIGNATURE.bind(*args, **kwargs)
        calls.append((args, kwargs))

    with patch("dapr_agents.observability.instrumentor.wrap_function_wrapper", spy):
        yield calls


@pytest.fixture
def instrumentor():
    inst = DaprAgentsInstrumentor()
    inst._tracer = trace.NoOpTracer()
    return inst


APPLY_METHODS = [
    "_apply_context_propagation_fix",
    "_apply_tool_wrappers",
    "_apply_workflow_wrappers",
    "_apply_llm_wrappers",
    "_apply_executor_wrappers",
]


@pytest.mark.parametrize("method", APPLY_METHODS)
def test_wrap_calls_pass_target_positionally(method, instrumentor, recorded_calls):
    with patch("dapr_agents.observability.instrumentor.logger") as log:
        getattr(instrumentor, method)()

    log.error.assert_not_called()
    assert recorded_calls, f"{method} made no wrap_function_wrapper calls"
    for args, kwargs in recorded_calls:
        assert "module" not in kwargs and "target" not in kwargs
        assert isinstance(args[0], str) and args[0]


def test_discover_chat_clients_includes_anthropic_litellm_mistral(instrumentor):
    """Every *ChatClient under dapr_agents.llm must be discovered (#882)."""
    from dapr_agents.llm.anthropic.chat import AnthropicChatClient
    from dapr_agents.llm.chat import ChatClientBase
    from dapr_agents.llm.litellm.chat import LiteLLMChatClient
    from dapr_agents.llm.mistral.chat import MistralChatClient

    discovered = instrumentor._discover_chat_clients(ChatClientBase)
    names = {cls.__name__ for cls, _ in discovered}
    modules = {module for _, module in discovered}

    assert AnthropicChatClient.__name__ in names
    assert LiteLLMChatClient.__name__ in names
    assert MistralChatClient.__name__ in names
    assert "dapr_agents.llm.anthropic.chat" in modules
    assert "dapr_agents.llm.litellm.chat" in modules
    assert "dapr_agents.llm.mistral.chat" in modules


def test_apply_llm_wrappers_instruments_anthropic_generate(
    instrumentor, recorded_calls
):
    """AnthropicChatClient.generate must be wrapped so LLM spans are emitted."""
    with patch("dapr_agents.observability.instrumentor.logger") as log:
        instrumentor._apply_llm_wrappers()

    log.error.assert_not_called()
    # wrap_function_wrapper(module, name=..., wrapper=...) — name may be kw or positional
    wrapped = set()
    for args, kwargs in recorded_calls:
        module = args[0]
        name = kwargs.get("name")
        if name is None and len(args) > 1:
            name = args[1]
        wrapped.add((module, name))
    assert (
        "dapr_agents.llm.anthropic.chat",
        "AnthropicChatClient.generate",
    ) in wrapped
    assert (
        "dapr_agents.llm.litellm.chat",
        "LiteLLMChatClient.generate",
    ) in wrapped
    assert (
        "dapr_agents.llm.mistral.chat",
        "MistralChatClient.generate",
    ) in wrapped
