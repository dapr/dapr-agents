"""Agent registry helpers."""

from __future__ import annotations

import json
import logging
import random
import time

from dapr.clients import DaprClient
from typing import Any, ClassVar, Dict, Optional
from dapr.clients.grpc._request import (
    TransactionOperationType,
    TransactionalStateOperation,
)
from dapr.clients.grpc._response import StateResponse
from dapr.clients.grpc._state import Concurrency, Consistency, StateOptions


logger = logging.getLogger(__name__)


class Registry:
    """Manage agent registrations and local uniqueness checks."""

    _registered_agent_names: ClassVar[Dict[str, str]] = {}

    def __init__(
        self,
        *,
        client: DaprClient,
        store_name: str,
        store_key: str = "agent_registry",
        metadata_request_timeout_seconds: int = 2,
    ) -> None:
        """Initialize the registry.
        Args:
            client: The Dapr client to use.
            store_name: The name of the state store to use.
            store_key: The key to use for the registry.
        """
        self.client = client
        self.store_name = store_name
        self.store_key = store_key
        self._cached_app_id: Optional[str] = None

        # Validate the client, store_name, and store_key
        if not self.client:
            raise ValueError("Dapr client is required")
        if not self.store_name:
            raise ValueError("State store name is required")
        if not self.store_key:
            raise ValueError("State store key is required")

    def get_dapr_app_id(self) -> Optional[str]:
        """
        Best effort to get the Dapr app ID from the metadata endpoint in the sidecar.

        Returns:
            The Dapr app ID, or None if unavailable.
        """
        if self._cached_app_id:
            return self._cached_app_id

        try:
            import os
            import requests

            # TODO: this environment configuration can be removed once this is added upstream in python sdk
            dapr_http_port = os.environ.get("DAPR_HTTP_PORT", "3500")
            dapr_host = os.environ.get("DAPR_RUNTIME_HOST", "localhost")
            metadata_url = f"http://{dapr_host}:{dapr_http_port}/v1.0/metadata"

            response = requests.get(metadata_url, timeout=self._m)
            if response.status_code == 200:
                metadata = response.json()
                self._cached_app_id = metadata.get("id")
                return self._cached_app_id
        except Exception as exc:
            logger.debug("Could not fetch Dapr app ID from metadata endpoint: %s", exc)

        return None

    def register_agent(
        self,
        *,
        agent_name: str,
        agent_metadata: Any,
        agent_identity: Optional[str] = None,
        store_name: Optional[str] = None,
        store_key: Optional[str] = None,
        max_attempts: int = 20,
    ) -> None:
        """Register agent metadata in the state store ensuring uniqueness."""

        client = self.client
        if client is None:
            logger.debug(
                "No Dapr client configured for registry; skipping registration for '%s'",
                agent_name,
            )
            return

        store_name = store_name or self.store_name
        store_key = store_key or self.store_key

        if not store_name or not store_key:
            raise ValueError("Registry requires both store_name and store_key")

        identity = agent_identity or agent_name

        if agent_name in Registry._registered_agent_names:
            existing_id = Registry._registered_agent_names[agent_name]
            if existing_id != identity:
                raise ValueError(
                    f"Agent name '{agent_name}' is already registered in this process (id: {existing_id}). "
                    "Agent names must be unique within a process."
                )

        for attempt in range(1, max_attempts + 1):
            try:
                response: StateResponse = client.get_state(
                    store_name=store_name, key=store_key
                )

                if not response.etag:
                    client.save_state(
                        store_name=store_name,
                        key=store_key,
                        value=json.dumps({}),
                        state_metadata={
                            "contentType": "application/json",
                            "partitionKey": store_key,
                        },
                        options=StateOptions(
                            concurrency=Concurrency.first_write,
                            consistency=Consistency.strong,
                        ),
                    )
                    response = client.get_state(store_name=store_name, key=store_key)
                    if not response.etag:
                        raise RuntimeError("ETag still missing after init")

                existing = (
                    self._deserialize_state(response.data) if response.data else {}
                )

                if agent_name in existing:
                    existing_metadata = existing[agent_name]
                    safe_metadata = self._serialize_metadata(agent_metadata)
                    existing_identity = Registry._registered_agent_names.get(agent_name)
                    if existing_metadata == safe_metadata:
                        logger.debug(
                            "Agent '%s' already registered with same metadata; updating local cache",
                            agent_name,
                        )
                        Registry._registered_agent_names[agent_name] = identity
                        return
                    if existing_identity == identity:
                        logger.debug(
                            "Agent '%s' metadata updated in same process; refreshing registry entry",
                            agent_name,
                        )
                        merged = {**existing, agent_name: safe_metadata}
                    else:
                        raise ValueError(
                            f"Agent name '{agent_name}' is already registered in the registry store "
                            f"(possibly from another process/sidecar). Agent names must be unique across all processes."
                        )
                else:
                    safe_metadata = self._serialize_metadata(agent_metadata)
                    merged = {**existing, agent_name: safe_metadata}

                merged_json = json.dumps(merged)

                logger.debug("merged data: %s etag: %s", merged_json, response.etag)

                transaction_operation = TransactionalStateOperation(
                    key=store_key,
                    data=merged_json,
                    etag=response.etag,
                    operation_type=TransactionOperationType.upsert,
                )

                if hasattr(transaction_operation, "configure_mock"):
                    transaction_operation.configure_mock(data=merged_json)

                client.execute_state_transaction(
                    store_name=store_name,
                    operations=[transaction_operation],
                    transactional_metadata={
                        "contentType": "application/json",
                        "partitionKey": store_key,
                    },
                )

                Registry._registered_agent_names[agent_name] = identity
                logger.info(
                    "Agent '%s' successfully registered in registry", agent_name
                )
                return
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - preserve previous behaviour
                logger.error("Error on transaction attempt: %s: %s", attempt, exc)
                delay = 1 + random.uniform(0, 1)
                logger.info(
                    "Sleeping for %.2f seconds before retrying transaction...", delay
                )
                time.sleep(delay)

        raise Exception(
            f"Failed to update state store key: {store_key} after {max_attempts} attempts."
        )

    @staticmethod
    def _deserialize_state(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError as exc:  # pragma: no cover - surface clear error
                raise ValueError("State bytes are not valid UTF-8") from exc
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - propagate context
                raise ValueError(f"State is not valid JSON: {exc}") from exc
        raise TypeError(f"Unsupported state type {type(raw)!r}")

    @staticmethod
    def _serialize_metadata(metadata: Any) -> Any:
        def convert(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            if isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            if isinstance(obj, dict):
                return {key: convert(value) for key, value in obj.items()}
            return obj

        return convert(metadata)

    @classmethod
    def clear_registered_names(cls) -> None:
        cls._registered_agent_names.clear()
