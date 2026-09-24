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
