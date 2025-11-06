from __future__ import annotations

from typing import Any, Dict, Optional
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

    def custom_container(model: BaseModel) -> Optional[Dict[str, Any]]:
        return getattr(model, "instances", None)

    config = AgentStateConfig(
        store=store,
        entry_factory=custom_entry_factory,
        message_coercer=custom_message_coercer,
        entry_container_getter=custom_container,
    )

    config.ensure_bundle(DEFAULT_AGENT_WORKFLOW_BUNDLE)
    bundle = config.get_state_model_bundle()

    assert bundle.state_model_cls is DEFAULT_AGENT_WORKFLOW_BUNDLE.state_model_cls
    assert bundle.message_model_cls is DEFAULT_AGENT_WORKFLOW_BUNDLE.message_model_cls
    assert bundle.entry_factory is custom_entry_factory
    assert bundle.message_coercer is custom_message_coercer
    assert bundle.entry_container_getter is custom_container


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

    class OtherState(BaseModel):
        value: int = 0

    class OtherMessage(BaseModel):
        role: str = "assistant"

    other_bundle = StateModelBundle(
        state_model_cls=OtherState,
        message_model_cls=OtherMessage,
    )

    try:
        config.ensure_bundle(other_bundle)
    except RuntimeError as exc:
        assert "Cannot inject" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when injecting mismatched bundle")
