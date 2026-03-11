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

import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions

from pydantic import BaseModel, ValidationError

from dapr_agents.agents.configs import (
    AgentMetadataSchema,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    DEFAULT_AGENT_WORKFLOW_BUNDLE,
    WorkflowGrpcOptions,
    StateModelBundle,
)


logger = logging.getLogger(__name__)

# Registry index key for the list of registered agent names
_REGISTRY_AGENTS_KEY = "agents"


class DaprInfra:
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
        override_state_key = state.state_key_prefix if state else None
        _normalized_name = self.name.replace(" ", "-").lower()
        self.state_key_prefix = override_state_key or f"{_normalized_name}:_workflow"

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
                "No state bundle for %s; using default agent workflow entry schema",
                self.name,
            )
            bundle = DEFAULT_AGENT_WORKFLOW_BUNDLE

        self._entry_model_cls = bundle.entry_model_cls
        self._message_model_cls = bundle.message_model_cls
        self._entry_factory = bundle.entry_factory
        self._message_coercer = bundle.message_coercer

        self._state_default_model: BaseModel = self._default_entry_model()
        self._state_model: BaseModel = self._state_default_model.model_copy(deep=True)
        self._last_etag: Optional[str] = None

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

    def load_state(self, workflow_instance_id: str) -> None:
        """
        Load the durable workflow state snapshot into memory.

        If no state store is configured, resets the in-memory model to defaults.
        """
        if not self.state_store:
            logger.debug("No state store configured; using in-memory state only.")
            self._state_model = self._initial_state_model()
            return

        if not workflow_instance_id:
            raise ValueError(
                "workflow_instance_id must be provided to load workflow state"
            )

        key = f"{self.state_key_prefix}_{workflow_instance_id}".lower()
        snapshot = self.state_store.load(
            key=key,
            default=self._initial_state(),
        )
        try:
            if isinstance(snapshot, dict):
                self._state_model = self._entry_model_cls.model_validate(snapshot)
            else:
                raise TypeError(f"Unexpected state snapshot type {type(snapshot)}")
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Invalid workflow state encountered (%s); resetting to defaults.", exc
            )
            self._state_model = self._initial_state_model()

    def get_state(self, workflow_instance_id: str) -> BaseModel:
        """
        Get the workflow state for a given workflow instance ID (read + set in-memory).

        Loads the entry from the store, validates it as the bundle's entry Pydantic model,
        sets it as the current in-memory state (so a subsequent save_state persists it),
        and returns it. Callers should mutate the returned model and then call save_state.

        The etag is cached so that a subsequent save_state can skip an extra load.

        Args:
            workflow_instance_id: The ID of the workflow instance to get the state for.

        Returns:
            The workflow entry model (e.g. AgentWorkflowEntry, LLMWorkflowEntry) for that instance.
        """
        if not self.state_store:
            logger.debug(
                "No state store configured; returning current in-memory state."
            )
            return self._state_model

        key = f"{self.state_key_prefix}_{workflow_instance_id}".lower()
        meta = self._state_metadata_for_key(key)
        snapshot, etag = self.state_store.load_with_etag(
            key=key,
            default=self._initial_state(),
            state_metadata=meta,
        )
        self._last_etag = etag
        try:
            if isinstance(snapshot, dict):
                entry = self._entry_model_cls.model_validate(snapshot)
                self._state_model = entry
                return entry
            raise TypeError(f"Unexpected state snapshot type {type(snapshot)}")
        except (ValidationError, TypeError) as exc:
            logger.warning(
                "Invalid workflow state encountered (%s); returning default entry.", exc
            )
            default = self._initial_state_model()
            self._state_model = default
            return default

    def save_state(self, workflow_instance_id: str) -> None:
        """
        Persist the current workflow state with optimistic concurrency.

        No-op when no state store is configured. Uses load_with_etag + save(etag=...)
        with a short retry loop to avoid lost updates under contention.
        """
        if not self.state_store:
            logger.debug("No state store configured; skipping state persistence.")
            return

        if not workflow_instance_id:
            raise ValueError(
                "workflow_instance_id must be provided to save workflow state"
            )

        key = f"{self.state_key_prefix}_{workflow_instance_id}".lower()
        meta = self._state_metadata_for_key(key)
        attempts = max(1, min(self._max_etag_attempts, 10))

        # Use the cached etag from a prior get_state when available to avoid
        # an extra round-trip.  Falls back to load_with_etag on the first
        # attempt when no cached etag exists, and always on retries.
        etag = self._last_etag
        self._last_etag = None  # consume; stale after save

        if etag is None:
            # No cached etag — ensure the document exists so we get one.
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
                    _, etag = self.state_store.load_with_etag(
                        key=key,
                        default=self._initial_state(),
                        state_metadata=meta,
                    )
            except Exception:
                logger.exception(
                    "Failed to initialize state document for key '%s'.", key
                )

        for attempt in range(1, attempts + 1):
            try:
                if etag is None:
                    # Shouldn't happen normally, but recover gracefully.
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
                # Refresh etag for next retry.
                etag = None
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

    def purge_state(self, workflow_instance_id: str) -> None:
        """
        Permanently delete workflow state for the given instance from the state store.

        No-op when no state store is configured.  Failures are logged as warnings
        and not re-raised so that callers can continue with other cleanup steps.

        Note: This method only removes workflow state.  To also purge long-term
        conversation memory use AgentBase.purge() / DurableAgent.purge(),
        which coordinate both workflow-state and memory cleanup in a single call.

        Args:
            workflow_instance_id: Workflow instance id whose state should be removed.
        """
        if not self.state_store:
            logger.debug("No state store configured; skipping purge_state.")
            return

        if not workflow_instance_id:
            raise ValueError(
                "workflow_instance_id must be provided to purge workflow state"
            )

        key = f"{self.state_key_prefix}_{workflow_instance_id}".lower()
        meta = self._state_metadata_for_key(key)
        try:
            self.state_store.delete(key=key, state_metadata=meta)
            logger.info(
                "Purged workflow state for instance_id=%s", workflow_instance_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to purge state for instance_id=%s: %s",
                workflow_instance_id,
                exc,
            )

    def _default_entry_model(self) -> BaseModel:
        """Return a default workflow entry model (one-key-per-instance)."""
        if self._entry_factory is not None:
            return self._entry_factory(
                instance_id="",
                input_value="",
                triggering_workflow_instance_id=None,
                start_time=datetime.now(timezone.utc),
            )
        # Fallback: minimal entry from schema (agent entry has no required fields).
        fields = self._entry_model_cls.model_fields
        kwargs: Dict[str, Any] = {}
        if "input" in fields:
            kwargs["input"] = ""
        return self._entry_model_cls(**kwargs)

    def _initial_state(self) -> Dict[str, Any]:
        """Return a deep-copied default state as a plain dict (entry shape)."""
        return self._state_default_model.model_copy(deep=True).model_dump(mode="json")

    def _initial_state_model(self) -> BaseModel:
        """Return a deep-copied default state model (entry)."""
        return self._state_default_model.model_copy(deep=True)

    def sync_system_messages(
        self,
        instance_id: str,
        all_messages: Sequence[Dict[str, Any]],
        *,
        entry: Optional[BaseModel] = None,
    ) -> None:
        """
        Synchronize system messages into the workflow state for a given instance.

        Uses `message_coercer` or `message_model_cls` to construct message entries.

        Args:
            instance_id: Workflow instance identifier.
            all_messages: Full (system/user/assistant) list; only 'system' are synced.
            entry: Pre-fetched state entry; when provided, skips the internal get_state call.
        """
        if entry is None:
            try:
                entry = self.get_state(instance_id)
            except Exception:
                logger.exception(
                    f"Failed to get workflow state for instance_id: {instance_id}"
                )
                raise
        if entry is None:
            try:
                entry = self.get_state(instance_id)
            except Exception:
                logger.exception(
                    f"Failed to get workflow state for instance_id: {instance_id}"
                )
                raise

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
            entry.system_messages = new_models

        # De-duplicate in entry.messages if that field exists
        if hasattr(entry, "messages"):
            filtered = [
                m
                for m in getattr(entry, "messages")
                if getattr(m, "role", None) != "system"
            ]
            entry.messages = filtered
            # Update last_message to point to the last non-system message
            # This ensures last_message always reflects the actual last message in the filtered list
            if hasattr(entry, "last_message"):
                entry.last_message = filtered[-1] if filtered else None

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
        metadata: Optional[AgentMetadataSchema] = None,
        team: Optional[str] = None,
    ) -> None:
        """
        Register agent metadata in the team registry.

        Two-step operation:
        1. Save per-agent key (simple overwrite, no read-modify-write).
        2. Update index with ETag-protected retry loop (add agent name to list).

        Args:
            metadata: Validated AgentMetadataSchema object
            team: Team override; falls back to configured default team
        """
        if self._registry is None or self.registry_state is None:
            logger.debug(
                "No registry configured; skipping registration for %s", self.name
            )
            return

        if metadata is None:
            logger.warning(
                "No metadata provided for registration of agent %s", self.name
            )
            return

        partition_meta = self._registry_partition_key(team)

        # Step 1: Save per-agent metadata key (contention-free overwrite)
        agent_key = self._agent_registry_key(self.name, team)
        metadata_dict = metadata.model_dump(mode="json")
        self.registry_state.save(
            key=agent_key,
            value=metadata_dict,
            state_metadata=partition_meta,
        )

        # Step 2: Add agent name to index (ETag-protected)
        # The retry loop handles both "create" (etag=None, first_write) and "update"
        # (etag!=None) cases atomically, so no separate initialization step is needed.
        index_key = self._team_registry_index_key(team)

        attempts = max(1, min(self._max_etag_attempts, 10))
        for attempt in range(1, attempts + 1):
            try:
                current_index, etag = self.registry_state.load_with_etag(
                    key=index_key,
                    default={_REGISTRY_AGENTS_KEY: []},
                    state_metadata=partition_meta,
                )
                agents_list = current_index.get(_REGISTRY_AGENTS_KEY, [])
                if self.name not in agents_list:
                    agents_list.append(self.name)
                    self.registry_state.save(
                        key=index_key,
                        value={_REGISTRY_AGENTS_KEY: agents_list},
                        etag=etag,
                        state_metadata=partition_meta,
                        state_options=self._save_options,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict updating registry index (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    index_key,
                    exc,
                )
                if attempt == attempts:
                    logger.exception(
                        "Failed to update registry index after %d attempts.", attempts
                    )
                    return
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

        logger.info(
            "Registered agent '%s' in team '%s' registry",
            self.name,
            self._effective_team(team),
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

        Reads the index to discover agent names, then bulk-fetches per-agent
        metadata keys.  Missing keys (stale index entries) are silently skipped.

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

        partition_meta = self._registry_partition_key(team)
        index_key = self._team_registry_index_key(team)
        try:
            index_data = self.registry_state.load(
                key=index_key,
                default={_REGISTRY_AGENTS_KEY: []},
                state_metadata=partition_meta,
            )
            agent_names = index_data.get(_REGISTRY_AGENTS_KEY, [])
            if not agent_names:
                logger.info("No agents found in registry index '%s'.", index_key)
                return {}

            # Build per-agent key list and bulk-fetch
            agent_keys = [self._agent_registry_key(name, team) for name in agent_names]
            bulk_results = self.registry_state.load_many(
                keys=agent_keys,
                state_metadata=partition_meta,
            )

            # Map results back to agent names, skipping missing keys
            agents_metadata: Dict[str, Any] = {}
            for name, key in zip(agent_names, agent_keys):
                meta = bulk_results.get(key)
                if meta is None:
                    continue
                agents_metadata[name] = meta

            filtered = {}
            for name, meta in agents_metadata.items():
                # Skip self if requested
                if exclude_self and name == self.name:
                    continue

                # Validate metadata structure - agent field must exist and be a dict
                agent_meta = meta.get("agent") if isinstance(meta, dict) else None
                if agent_meta is None or not isinstance(agent_meta, dict):
                    logger.error(
                        "Agent '%s' has invalid metadata structure: missing or invalid 'agent' field. "
                        "This indicates corrupted registry data. Metadata: %s",
                        name,
                        meta,
                    )
                    continue

                # Skip orchestrators if requested
                if exclude_orchestrator and agent_meta.get("orchestrator", False):
                    continue

                filtered[name] = meta
            return filtered
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to retrieve agents metadata: %s", exc, exc_info=True)
            raise RuntimeError(f"Error retrieving agents metadata: {str(exc)}") from exc

    def _remove_agent_entry(
        self,
        *,
        team: Optional[str],
        agent_name: str,
    ) -> None:
        """
        Delete a single agent record from the team registry.

        Two-step operation:
        1. Delete per-agent key (simple delete).
        2. Update index with ETag-protected retry loop (remove agent name from list).

        The per-agent key is deleted first so that if the index update fails,
        get_agents_metadata() gracefully skips the stale entry.

        Args:
            team: Team identifier.
            agent_name: Agent name (key).
        """
        partition_meta = self._registry_partition_key(team)

        # Step 1: Delete per-agent metadata key
        agent_key = self._agent_registry_key(agent_name, team)
        try:
            self.registry_state.delete(key=agent_key, state_metadata=partition_meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete per-agent key '%s': %s", agent_key, exc)

        # Step 2: Remove agent name from index (ETag-protected)
        index_key = self._team_registry_index_key(team)
        attempts = max(1, min(self._max_etag_attempts, 10))
        for attempt in range(1, attempts + 1):
            try:
                current_index, etag = self.registry_state.load_with_etag(
                    key=index_key,
                    default={_REGISTRY_AGENTS_KEY: []},
                    state_metadata=partition_meta,
                )
                agents_list = current_index.get(_REGISTRY_AGENTS_KEY, [])
                if agent_name not in agents_list:
                    break
                agents_list.remove(agent_name)
                self.registry_state.save(
                    key=index_key,
                    value={_REGISTRY_AGENTS_KEY: agents_list},
                    etag=etag,
                    state_metadata=partition_meta,
                    state_options=self._save_options,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict updating registry index (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    index_key,
                    exc,
                )
                if attempt == attempts:
                    logger.exception(
                        "Failed to update registry index after %d attempts.", attempts
                    )
                    return
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

        logger.info(
            "Deregistered agent '%s' from team '%s' registry",
            agent_name,
            self._effective_team(team),
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

    def _team_registry_index_key(self, team: Optional[str] = None) -> str:
        """Return the index key that lists all registered agent names for a team."""
        return f"{self._registry_prefix}{self._effective_team(team)}:_index"

    def _agent_registry_key(self, agent_name: str, team: Optional[str] = None) -> str:
        """Return the per-agent metadata key within a team."""
        return f"{self._registry_prefix}{self._effective_team(team)}:{agent_name}"

    def _registry_partition_key(self, team: Optional[str] = None) -> Dict[str, str]:
        """Return state metadata with a common partition key for all registry keys within a team."""
        meta = dict(self._base_metadata)
        meta["partitionKey"] = self._team_registry_key(team)
        return meta

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
        current, etag = self.registry_state.load_with_etag(
            key=key,
            default={},
            state_metadata=meta,
        )
        if etag is None:
            self.registry_state.save(
                key=key,
                value={},
                etag=None,
                state_metadata=meta,
                state_options=self._save_options,
            )

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
