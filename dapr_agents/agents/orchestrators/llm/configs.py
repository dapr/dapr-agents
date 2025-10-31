from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, MutableMapping, Optional, Type

from pydantic import BaseModel

from dapr_agents.agents.configs import (
    AgentStateConfig,
    EntryContainerGetter,
    EntryFactory,
    MessageCoercer,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from dapr_agents.agents.orchestrators.llm.state import (
    LLMWorkflowEntry,
    LLMWorkflowMessage,
    LLMWorkflowState,
)


# ---------- helpers (module-private) ----------

def _utcnow() -> datetime:
    """Timezone-aware now in UTC."""
    return datetime.now(timezone.utc)


def _maybe_aware(dt: Optional[datetime]) -> datetime:
    """Coerce naive datetimes to UTC."""
    if dt is None:
        return _utcnow()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _default_entry_factory(
    *,
    instance_id: str,
    input_value: Any,
    triggering_workflow_instance_id: Optional[str],
    start_time: Optional[datetime],
) -> LLMWorkflowEntry:
    """
    Create a baseline LLM workflow entry with non-null collections.
    Forward-compatible: sets optional ids only if model defines them.
    """
    ts = _maybe_aware(start_time)

    # Only populate optional ids if the model actually defines those fields
    opt: Dict[str, Any] = {}
    fields = getattr(LLMWorkflowEntry, "model_fields", {})
    if "workflow_instance_id" in fields:
        opt["workflow_instance_id"] = instance_id
    if "triggering_workflow_instance_id" in fields:
        opt["triggering_workflow_instance_id"] = triggering_workflow_instance_id

    return LLMWorkflowEntry(
        input=str(input_value or ""),
        output=None,
        start_time=ts,
        end_time=None,
        messages=[],
        last_message=None,
        plan=[],          # never None
        task_history=[],  # never None
        **opt,
    )


def _default_message_coercer(raw: Dict[str, Any]) -> LLMWorkflowMessage:
    """
    Coerce raw dicts into the LLM message model.
    - Whitelists known fields
    - Defaults role/content
    - Accepts either datetime or ISO8601 string timestamps
    """
    allowed = {"role", "content", "name", "id", "timestamp"}
    payload = {k: raw[k] for k in allowed if k in raw}

    # sensible defaults
    payload.setdefault("role", "system")

    content = payload.get("content", "")
    payload["content"] = content if isinstance(content, str) else str(content)

    # timestamp: accept str or datetime and coerce to aware datetime
    ts = payload.get("timestamp")
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            payload["timestamp"] = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            payload["timestamp"] = _utcnow()
    elif isinstance(ts, datetime):
        payload["timestamp"] = _maybe_aware(ts)
    else:
        payload["timestamp"] = _utcnow()

    return LLMWorkflowMessage(**payload)


def _default_entry_container_getter(model: BaseModel) -> Optional[MutableMapping[str, Any]]:
    """Return the container that maps instance_id -> entry (if present)."""
    return getattr(model, "instances", None)


# ---------- public config ----------

@dataclass
class LLMOrchestratorStateConfig(AgentStateConfig):
    """
    Drop-in state config for LLM orchestrators.

    Defaults:
      • state_model_cls = LLMWorkflowState
      • message_model_cls = LLMWorkflowMessage
      • entry_factory = _default_entry_factory (non-null plan/history; optional ids)
      • message_coercer = _default_message_coercer (defensive + UTC)
      • entry_container_getter = instances

    Only `store` is required:
        LLMOrchestratorStateConfig(store=StateStoreService(...))
    """
    # required
    store: StateStoreService = None  # type: ignore[assignment]

    # baked-in LLM defaults
    state_model_cls: Type[BaseModel] = LLMWorkflowState
    message_model_cls: Type[BaseModel] = LLMWorkflowMessage
    entry_factory: Optional[EntryFactory] = _default_entry_factory
    message_coercer: Optional[MessageCoercer] = _default_message_coercer
    entry_container_getter: Optional[EntryContainerGetter] = _default_entry_container_getter

    def __post_init__(self) -> None:
        """
        Ensure the base normalization runs with our defaults.
        - Validates model classes
        - Normalizes/validates `default_state` against LLMWorkflowState
        """
        super().__post_init__()