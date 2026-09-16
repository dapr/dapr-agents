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

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import Mock

from pydantic import BaseModel

from dapr_agents.agents.configs import (
    AgentStateConfig,
    DEFAULT_AGENT_WORKFLOW_BUNDLE,
    StateModelBundle,
)


def test_ensure_bundle_merges_user_hooks() -> None:
    store = Mock()
    store.store_name = "state"

    def custom_entry_factory(**_: Any) -> str:
        return "custom"

    def custom_message_coercer(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {**payload, "role": "system"}

    config = AgentStateConfig(
        store=store,
        entry_factory=custom_entry_factory,
        message_coercer=custom_message_coercer,
    )

    config.ensure_bundle(DEFAULT_AGENT_WORKFLOW_BUNDLE)
    bundle = config.get_state_model_bundle()

    assert bundle.entry_model_cls is DEFAULT_AGENT_WORKFLOW_BUNDLE.entry_model_cls
    assert bundle.message_model_cls is DEFAULT_AGENT_WORKFLOW_BUNDLE.message_model_cls
    assert bundle.entry_factory is custom_entry_factory
    assert bundle.message_coercer is custom_message_coercer


def test_ensure_bundle_is_idempotent() -> None:
    store = Mock()
    store.store_name = "state"
    config = AgentStateConfig(store=store)

    config.ensure_bundle(DEFAULT_AGENT_WORKFLOW_BUNDLE)
    # second injection with same bundle should be a no-op
    config.ensure_bundle(DEFAULT_AGENT_WORKFLOW_BUNDLE)


def test_ensure_bundle_rejects_mismatched_schema() -> None:
    store = Mock()
    store.store_name = "state"
    config = AgentStateConfig(store=store)
    config.ensure_bundle(DEFAULT_AGENT_WORKFLOW_BUNDLE)

    class OtherEntry(BaseModel):
        value: int = 0

    class OtherMessage(BaseModel):
        role: str = "assistant"

    other_bundle = StateModelBundle(
        entry_model_cls=OtherEntry,
        message_model_cls=OtherMessage,
    )

    try:
        config.ensure_bundle(other_bundle)
    except RuntimeError as exc:
        assert "Cannot inject" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when injecting mismatched bundle")
