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

from dapr_agents.agents.schemas import TriggerAction


def test_trigger_action_backwards_compat():
    t = TriggerAction(task="say hello")
    assert t.task == "say hello"
    assert t.workflow_instance_id is None
    assert t.caller_headers is None


def test_trigger_action_with_caller_headers():
    headers = {
        "Authorization": "Bearer eyJhbG...",
        "X-Request-Id": "req-c4d2e1f0",
        "traceparent": "00-4bf92f3577b34da6-a3ce929d0e0e4736-01",
    }
    t = TriggerAction(task="x", caller_headers=headers)
    assert t.caller_headers["Authorization"].startswith("Bearer ")
    assert t.caller_headers["X-Request-Id"] == "req-c4d2e1f0"


def test_trigger_action_serialization_roundtrip():
    t = TriggerAction(
        task="x",
        workflow_instance_id="wf-1",
        caller_headers={"Authorization": "Bearer x"},
    )
    payload = t.model_dump()
    t2 = TriggerAction.model_validate(payload)
    assert t2.task == "x"
    assert t2.workflow_instance_id == "wf-1"
    assert t2.caller_headers == {"Authorization": "Bearer x"}


def test_trigger_action_omitted_headers_serializes_as_none():
    t = TriggerAction(task="x")
    payload = t.model_dump()
    assert payload.get("caller_headers") is None


def test_trigger_action_empty_headers_dict():
    # Empty dict allowed; semantically equivalent to None for plugin behavior
    t = TriggerAction(task="x", caller_headers={})
    assert t.caller_headers == {}
