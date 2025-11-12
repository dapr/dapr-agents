from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions
from pydantic import BaseModel, ValidationError

from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    DEFAULT_AGENT_WORKFLOW_BUNDLE,
    WorkflowGrpcOptions,
    StateModelBundle,
)
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.storage.daprstores.stateservice import StateStoreError
from dapr_agents.types.workflow import DaprWorkflowStatus

logger = logging.getLogger(__name__)


class AgentComponents:
    """
    Thin infrastructure layer for agents/orchestrators.

    Handles:
    - Pub/Sub plumbing (topic names, bus name).
    - Durable workflow state (load/save, instance bootstrapping).
    - Team registry operations (list/register/mutate with optimistic concurrency).

    Higher-level concerns (prompting, memory, tools) should remain outside this class.
    """

    def __init__(
        self,
        *,
        name: str,
        pubsub: Optional[AgentPubSubConfig] = None,
        state: Optional[AgentStateConfig] = None,
        registry: Optional[AgentRegistryConfig] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
        max_etag_attempts: int = 10,
        workflow_grpc_options: Optional[WorkflowGrpcOptions] = None,
        default_bundle: Optional[StateModelBundle] = None,
    ) -> None:
        """
        Initialize component wiring.

        Args:
            name: Logical agent name; used for keys/topics when not overridden.
            pubsub: Dapr pub/sub configuration for this agent.
            state: Durable state (Dapr state store, key overrides, defaults, hooks).
            registry: Agent registry backing store and team settings.
            base_metadata: Base metadata for Dapr state operations.
            max_etag_attempts: Max optimistic-concurrency retries on registry mutations.
            default_bundle: Default state schema bundle (injected by agent/orchestrator class).
        """
        self.name = name
        self._workflow_grpc_options = workflow_grpc_options

        # -----------------------------
        # Pub/Sub configuration (copy)
        # -----------------------------
        self._pubsub: Optional[AgentPubSubConfig] = None
        if pubsub is not None:
            # Copy only what we need to avoid accidental external mutation.
            self._pubsub = AgentPubSubConfig(
                pubsub_name=pubsub.pubsub_name,
                agent_topic=pubsub.agent_topic or name,
                broadcast_topic=pubsub.broadcast_topic,
            )

        # -----------------------------
        # State configuration and model (flexible)
        # -----------------------------
        self._state = state
        self.state_store = state.store if state and state.store else None
        override_state_key = state.state_key if state else None
        self.state_key = override_state_key or f"{self.name}:workflow_state"

        bundle = None
        if state is not None:
            # Allow default_bundle to override the state's bundle. This enables
            # orchestrators and agents to share the same AgentStateConfig instance
            # while each using their own specialized state model schemas.
            if default_bundle is not None:
                state.ensure_bundle(default_bundle)
            try:
                bundle = state.get_state_model_bundle()
            except RuntimeError:
                bundle = None
        elif default_bundle is not None:
            bundle = default_bundle

        if bundle is None:
            logger.debug(
                "No state bundle for %s; using default AgentWorkflowState schema",
                self.name,
            )
            bundle = DEFAULT_AGENT_WORKFLOW_BUNDLE

        # I considered splitting into separate classes, but that would duplicate several lines
        # of infrastructure code (pub/sub, state operations, registry mutations). The current design
        # uses the Strategy Pattern to share infrastructure while maintaining type-safe schemas per
        # agent/orchestrator type. The "complexity" is just 5 lines of bundle extraction vs maintaining
        # duplicate codebases.
        self._state_model_cls = bundle.state_model_cls
        self._message_model_cls = bundle.message_model_cls
        self._entry_factory = bundle.entry_factory
        self._message_coercer = bundle.message_coercer
        self._entry_container_getter = bundle.entry_container_getter

        # Seed the default model from config or empty instance
        if state and state.default_state is not None:
            default_state_model = self._state_model_cls.model_validate(
                state.default_state
            )
        else:
            default_state_model = self._state_model_cls()
        self._state_default_model: BaseModel = default_state_model
        self._state_model: BaseModel = self._state_default_model.model_copy(deep=True)

        # -----------------------------
        # Registry configuration
        # -----------------------------
        self._registry = registry
        self.registry_state = registry.store if registry else None
        self._registry_prefix = "agents:"
        self._registry_team_override = (
            registry.team_name if registry and registry.team_name else "default"
        )

        # -----------------------------
        # Dapr save options & metadata
        # -----------------------------
        self._save_options = StateOptions(
            concurrency=Concurrency.first_write,
            consistency=Consistency.strong,
        )
        self._base_metadata = dict(base_metadata or {"contentType": "application/json"})
        self._max_etag_attempts = max_etag_attempts

    # ------------------------------------------------------------------
    # Pub/Sub helpers
    # ------------------------------------------------------------------
    @property
    def pubsub(self) -> Optional[AgentPubSubConfig]:
        """Return the configured pub/sub settings, if any."""
        return self._pubsub

    @property
    def message_bus_name(self) -> Optional[str]:
        """Return the Dapr pub/sub component name (bus), or None if no pubsub configured."""
        if not self._pubsub:
            return None
        return self._pubsub.pubsub_name

    @property
    def agent_topic_name(self) -> Optional[str]:
        """Return the per-agent topic name, or None if no pubsub configured."""
        if not self._pubsub:
            return None
        return self._pubsub.agent_topic or self.name

    @property
    def broadcast_topic_name(self) -> Optional[str]:
        """Return the broadcast topic name, if one was configured."""
        if not self._pubsub:
            return None
        return self._pubsub.broadcast_topic

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def workflow_state(self) -> BaseModel:
        """Return the in-memory workflow state model (customizable model)."""
        return self._state_model

    @property
    def workflow_grpc_options(self) -> Optional[WorkflowGrpcOptions]:
        """Return workflow gRPC tuning options if provided."""
        return self._workflow_grpc_options

    @property
    def state(self) -> Dict[str, Any]:
        """Return the workflow state as a JSON-serializable dict."""
        return self._state_model.model_dump(mode="json")

    def load_state(self) -> None:
        """
        Load the durable workflow state snapshot into memory.

        If no state store is configured, resets the in-memory model to defaults.
        """
        if not self.state_store:
            logger.debug("No state store configured; using in-memory state only.")
            self._state_model = self._initial_state_model()
            return

        snapshot = self.state_store.load(
            key=self.state_key,
            default=self._initial_state(),
        )
        try:
            if isinstance(snapshot, dict):
                self._state_model = self._state_model_cls.model_validate(snapshot)
            else:
                raise TypeError(f"Unexpected state snapshot type {type(snapshot)}")
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Invalid workflow state encountered (%s); resetting to defaults.", exc
            )
            self._state_model = self._initial_state_model()

    def save_state(self) -> None:
        """
        Persist the current workflow state with optimistic concurrency.

        No-op when no state store is configured. Uses load_with_etag + save(etag=...)
        with a short retry loop to avoid lost updates under contention.
        """
        if not self.state_store:
            logger.debug("No state store configured; skipping state persistence.")
            return

        key = self.state_key
        meta = self._state_metadata_for_key(key)
        attempts = max(1, min(self._max_etag_attempts, 10))

        # Ensure the state document exists so we can get a concrete ETag.
        try:
            current, etag = self.state_store.load_with_etag(
                key=key,
                default=self._initial_state(),
                state_metadata=meta,
            )
            if etag is None:
                # Initialize to get an etag
                self.state_store.save(
                    key=key,
                    value=current if isinstance(current, dict) else self.state,
                    etag=None,
                    state_metadata=meta,
                    state_options=self._save_options,
                )
        except Exception:
            logger.exception("Failed to initialize state document for key '%s'.", key)
            # Best-effort attempt to proceed; if this fails below, we'll log again.

        for attempt in range(1, attempts + 1):
            try:
                _, etag = self.state_store.load_with_etag(
                    key=key,
                    default=self._initial_state(),
                    state_metadata=meta,
                )
                self.state_store.save(
                    key=key,
                    value=self.state,
                    etag=etag,
                    state_metadata=meta,
                    state_options=self._save_options,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict during workflow state save (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    key,
                    exc,
                )
                if attempt == attempts:
                    logger.exception(
                        "Failed to persist agent state after %d attempts.", attempts
                    )
                    return
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

    def _initial_state(self) -> Dict[str, Any]:
        """Return a deep-copied default state as a plain dict."""
        return self._state_default_model.model_copy(deep=True).model_dump(mode="json")

    def _initial_state_model(self) -> BaseModel:
        """Return a deep-copied default state model."""
        return self._state_default_model.model_copy(deep=True)

    def ensure_instance_exists(
        self,
        *,
        instance_id: str,
        input_value: Any,
        triggering_workflow_instance_id: Optional[str],
        time: Optional[datetime] = None,
    ) -> None:
        """
        Ensure a workflow instance entry exists in the state model.

        Uses a pluggable `entry_factory` when provided. If absent, falls back to a
        best-effort default that assumes an `instances` dict on the root model.

        Args:
            instance_id: Unique workflow instance identifier.
            input_value: Input payload used to start the workflow.
            triggering_workflow_instance_id: Parent workflow instance id, if any.
            time: Optional start time (defaults to now, UTC).

        Raises:
            RuntimeError: If a custom entry factory raises and is not handled.
        """
        container = self._get_entry_container()
        if container is None:
            # No instances concept; nothing to do.
            return
        if instance_id in container:
            return

        start_time = self._coerce_datetime(time)

        if self._entry_factory is not None:
            entry = self._entry_factory(
                instance_id=instance_id,
                input_value=input_value,
                triggering_workflow_instance_id=triggering_workflow_instance_id,
                start_time=start_time,
            )
        else:
            # Default (legacy) AgentWorkflowEntry-compatible record
            entry = AgentWorkflowEntry(
                input_value=str(input_value),
                workflow_instance_id=instance_id,
                triggering_workflow_instance_id=triggering_workflow_instance_id,
                workflow_name=None,
                session_id=None,
                start_time=start_time,
                status=DaprWorkflowStatus.RUNNING.value,
            )
        container[instance_id] = entry

    def sync_system_messages(
        self,
        instance_id: str,
        all_messages: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Synchronize system messages into the workflow state for a given instance.

        Uses `message_coercer` or `message_model_cls` to construct message entries.

        Args:
            instance_id: Workflow instance identifier.
            all_messages: Full (system/user/assistant) list; only 'system' are synced.
        """
        container = self._get_entry_container()
        if container is None:
            return
        entry = container.get(instance_id)
        if entry is None:
            return

        system_messages = [m for m in all_messages if m.get("role") == "system"]
        if not system_messages:
            return

        existing = list(getattr(entry, "system_messages", []) or [])
        existing_sig = [
            (getattr(m, "content", None), getattr(m, "name", None)) for m in existing
        ]
        new_sig = [(m.get("content"), m.get("name")) for m in system_messages]
        if existing_sig == new_sig:
            return

        # Build new models
        if self._message_coercer:
            new_models = [self._message_coercer(m) for m in system_messages]
        else:
            new_models = [
                self._message_dict_to_message_model(m) for m in system_messages
            ]

        # Assign back if the field exists; otherwise, skip
        if hasattr(entry, "system_messages"):
            entry.system_messages = new_models  # type: ignore[attr-defined]

        # De-duplicate in entry.messages if that field exists
        if hasattr(entry, "messages"):
            filtered = [
                m
                for m in getattr(entry, "messages")
                if getattr(m, "role", None) != "system"
            ]
            entry.messages = filtered  # type: ignore[attr-defined]
            # Fix last_message if applicable
            if (
                getattr(entry, "last_message", None) is not None
                and getattr(entry.last_message, "role", None) == "system"
            ):
                non_system = [
                    m
                    for m in getattr(entry, "messages")
                    if getattr(m, "role", None) != "system"
                ]
                entry.last_message = non_system[-1] if non_system else None  # type: ignore[attr-defined]

    def _message_dict_to_message_model(self, message: Dict[str, Any]) -> Any:
        """
        Convert a dict into the configured message model.

        Falls back to returning the raw dict if instantiation fails (to avoid hard
        failures with custom models). Logs a warning the first time a shape mismatch
        is observed to help with debugging template drift.
        """
        allowed = {
            "role",
            "content",
            "name",
            "tool_calls",
            "function_call",
            "tool_call_id",
            "id",
        }
        payload = {k: message[k] for k in allowed if k in message}
        payload.setdefault("role", "system")
        payload.setdefault("content", "")

        try:
            return self._message_model_cls(**payload)
        except Exception as exc:  # noqa: BLE001
            # Keep noisy logs under control by summarizing the mismatch.
            try:
                role = payload.get("role")
                name = payload.get("name")
                msg_id = payload.get("id") or payload.get("tool_call_id")
                logger.warning(
                    "Message coercion failed for role=%r name=%r id=%r with %s; keeping raw payload.",
                    role,
                    name,
                    msg_id,
                    type(exc).__name__,
                )
            except Exception:
                # Don't let logging fail the fallback
                pass
            return payload

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def register_agentic_system(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        team: Optional[str] = None,
    ) -> None:
        """
        Upsert this agent's metadata in the team registry.

        Args:
            metadata: Additional metadata to store for this agent.
            team: Team override; falls back to configured default team.
        """
        if not self.registry_state:
            logger.debug(
                "No registry configured; skipping registration for %s", self.name
            )
            return

        payload = dict(metadata or {})
        payload.setdefault("name", self.name)
        payload.setdefault("team", self._effective_team(team))

        if self._pubsub is not None:
            payload.setdefault("topic_name", self.agent_topic_name)
            payload.setdefault("pubsub_name", self.message_bus_name)
            if self.broadcast_topic_name:
                payload.setdefault("broadcast_topic", self.broadcast_topic_name)

        self._upsert_agent_entry(
            team=self._effective_team(team),
            agent_name=self.name,
            agent_metadata=payload,
        )

    def deregister_agentic_system(self, *, team: Optional[str] = None) -> None:
        """
        Remove this agent from the team registry.

        Args:
            team: Team override; falls back to configured default team.
        """
        if not self.registry_state:
            return
        self._remove_agent_entry(team=self._effective_team(team), agent_name=self.name)

    def get_agents_metadata(
        self,
        *,
        exclude_self: bool = True,
        exclude_orchestrator: bool = False,
        team: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load and optionally filter all agents registered for a team.

        Args:
            exclude_self: If True, omit this agent from results.
            exclude_orchestrator: If True, omit agents with orchestrator=True.
            team: Team override; falls back to configured default team.

        Returns:
            Mapping of agent name to metadata.

        Raises:
            RuntimeError: When no registry is configured or load fails.
        """
        if not self.registry_state:
            raise RuntimeError("registry_state must be provided to use agent registry")

        key = self._team_registry_key(team)
        try:
            agents_metadata = self.registry_state.load(
                key=key,
                default={},
                state_metadata=self._state_metadata_for_key(key),
            )
            if not agents_metadata:
                logger.info("No agents found in registry key '%s'.", key)
                return {}

            filtered = {
                name: meta
                for name, meta in agents_metadata.items()
                if not (exclude_self and name == self.name)
                and not (exclude_orchestrator and meta.get("orchestrator", False))
            }
            return filtered
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to retrieve agents metadata: %s", exc, exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(exc)}") from exc

    def _mutate_registry_entry(
        self,
        *,
        team: Optional[str],
        mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
        max_attempts: Optional[int] = None,
    ) -> None:
        """
        Apply a mutation to the team registry with optimistic concurrency.

        Args:
            team: Team identifier.
            mutator: Function that returns the updated registry dict (or None for no-op).
            max_attempts: Override for concurrency retries; defaults to init value.

        Raises:
            StateStoreError: If the mutation fails after retries due to contention.
        """
        if not self.registry_state:
            raise RuntimeError(
                "registry_state must be provided to mutate the agent registry"
            )

        key = self._team_registry_key(team)
        meta = self._state_metadata_for_key(key)
        attempts = max_attempts or self._max_etag_attempts

        self._ensure_registry_initialized(key=key, meta=meta)

        for attempt in range(1, attempts + 1):
            try:
                current, etag = self.registry_state.load_with_etag(
                    key=key,
                    default={},
                    state_metadata=meta,
                )
                if not isinstance(current, dict):
                    current = {}

                updated = mutator(dict(current))
                if updated is None:
                    return

                self.registry_state.save(
                    key=key,
                    value=updated,
                    etag=etag,
                    state_metadata=meta,
                    state_options=self._save_options,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict during registry mutation (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    key,
                    exc,
                )
                if attempt == attempts:
                    raise StateStoreError(
                        f"Failed to mutate agent registry key '{key}' after {attempts} attempts."
                    ) from exc
                # Jittered backoff to reduce thundering herd during contention.
                time.sleep(min(1.0 * attempt, 3.0) * (1 + random.uniform(0, 0.25)))

    def _upsert_agent_entry(
        self,
        *,
        team: Optional[str],
        agent_name: str,
        agent_metadata: Dict[str, Any],
        max_attempts: Optional[int] = None,
    ) -> None:
        """
        Insert/update a single agent record in the team registry.

        Args:
            team: Team identifier.
            agent_name: Agent name (key).
            agent_metadata: Metadata value to write.
            max_attempts: Override retry attempts.
        """

        def mutator(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if current.get(agent_name) == agent_metadata:
                return None
            current[agent_name] = agent_metadata
            return current

        self._mutate_registry_entry(
            team=team,
            mutator=mutator,
            max_attempts=max_attempts,
        )

    def _remove_agent_entry(
        self,
        *,
        team: Optional[str],
        agent_name: str,
        max_attempts: Optional[int] = None,
    ) -> None:
        """
        Delete a single agent record from the team registry.

        Args:
            team: Team identifier.
            agent_name: Agent name (key).
            max_attempts: Override retry attempts.
        """

        def mutator(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if agent_name not in current:
                return None
            del current[agent_name]
            return current

        self._mutate_registry_entry(
            team=team,
            mutator=mutator,
            max_attempts=max_attempts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _effective_team(self, team: Optional[str] = None) -> str:
        """Compute the effective team name from override or defaults."""
        return team or self._registry_team_override or "default"

    def effective_team(self, team: Optional[str] = None) -> str:
        """Public alias for _effective_team, useful in templates/callers."""
        return self._effective_team(team)

    def _team_registry_key(self, team: Optional[str] = None) -> str:
        """Return the registry document key for a team."""
        return f"{self._registry_prefix}{self._effective_team(team)}"

    def _state_metadata_for_key(self, key: str) -> Dict[str, str]:
        """Return Dapr state metadata including partition key."""
        meta = dict(self._base_metadata)
        meta["partitionKey"] = key
        return meta

    def _ensure_registry_initialized(self, *, key: str, meta: Dict[str, str]) -> None:
        """
        Ensure a registry document exists to create an ETag for concurrency control.

        Args:
            key: Registry document key.
            meta: Dapr state metadata to use for the operation.
        """
        current, etag = self.registry_state.load_with_etag(  # type: ignore[union-attr]
            key=key,
            default={},
            state_metadata=meta,
        )
        if etag is None:
            self.registry_state.save(  # type: ignore[union-attr]
                key=key,
                value={},
                etag=None,
                state_metadata=meta,
                state_options=self._save_options,
            )

    def _get_entry_container(self) -> Optional[dict]:
        """
        Return the container mapping for workflow entries, if any.

        Returns:
            A mutable mapping (e.g., dict) of instance_id -> entry, or None if
            the underlying state model does not expose such a container.

        Notes:
            Prefer a caller-provided hook via `AgentStateConfig.entry_container_getter`.
            Falls back to `model.instances` for legacy/default shapes.
        """
        if self._entry_container_getter:
            return self._entry_container_getter(self._state_model)
        return getattr(self._state_model, "instances", None)

    @staticmethod
    def _coerce_datetime(value: Optional[Any]) -> datetime:
        """
        Coerce strings/None to a timezone-aware UTC datetime.

        Args:
            value: Source value (datetime | str | None).

        Returns:
            A timezone-aware UTC datetime. If a naive datetime is provided, UTC is assumed.
        """
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)
