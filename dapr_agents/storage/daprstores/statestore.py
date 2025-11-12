from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dapr.clients import DaprClient
from dapr.clients.grpc._response import (
    BulkStateItem,
    BulkStatesResponse,
    QueryResponse,
    StateResponse,
)
from dapr.clients.grpc._state import StateItem, StateOptions

from dapr_agents.storage.daprstores.base import DaprStoreBase


def _coerce_state_options(
    state_options: Optional[Union[StateOptions, Dict[str, Any]]],
) -> Optional[StateOptions]:
    """
    Convert a dict of state options into a `StateOptions` instance, or pass
    through an existing `StateOptions`.

    Args:
        state_options: None, a dict matching `StateOptions` fields, or a `StateOptions`.

    Returns:
        A `StateOptions` instance or None.
    """
    if state_options is None:
        return None

    # Prefer explicit dict detection first; newer typing helpers may wrap StateOptions
    # in `typing.NewType`/Union-style aliases that `isinstance` cannot handle.
    if isinstance(state_options, dict):
        return StateOptions(**state_options)

    # Fallback: treat any object exposing the expected attributes as StateOptions-like.
    if hasattr(state_options, "consistency") and hasattr(state_options, "concurrency"):
        return state_options  # type: ignore[return-value]

    # When annotations or typing aliases wrap the class, fall back to constructing one.
    return StateOptions(**dict(state_options))  # type: ignore[arg-type]


class DaprStateStore(DaprStoreBase):
    """
    Thin wrapper around Dapr state APIs returning raw gRPC response types.

    This class intentionally avoids JSON coercion, validation, prefixing, retries,
    and mirroring. If you want those conveniences, use `StateStoreService`.
    """

    def get_state(
        self,
        key: str,
        *,
        state_metadata: Optional[Dict[str, str]] = None,
    ) -> StateResponse:
        """
        Retrieve a single state item.

        Args:
            key: Key to fetch (as stored in the Dapr component).
            state_metadata: Optional Dapr metadata for the request.

        Returns:
            `StateResponse` containing bytes payload, etag, and metadata.
        """
        with DaprClient() as client:
            return client.get_state(
                store_name=self.store_name,
                key=key,
                state_metadata=state_metadata,
            )

    def try_get_state(
        self,
        key: str,
        *,
        state_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Optional[dict]]:
        """
        Attempt to get a JSON-encoded state item and decode it into a dict.

        Args:
            key: Key to fetch.
            state_metadata: Optional Dapr metadata.

        Returns:
            (exists, payload_dict_or_none)
        """
        response = self.get_state(
            key=key,
            state_metadata=state_metadata,
        )
        if response and response.data:
            return True, response.json()
        return False, None

    def get_bulk_state(
        self,
        keys: List[str],
        *,
        parallelism: int = 1,
        states_metadata: Optional[Dict[str, str]] = None,
    ) -> List[BulkStateItem]:
        """
        Retrieve multiple keys in one call.

        Args:
            keys: Keys to fetch.
            parallelism: How many to fetch in parallel (backend dependent).
            states_metadata: Optional bulk metadata.

        Returns:
            List of `BulkStateItem`. Items with missing keys may have empty data.
        """
        with DaprClient() as client:
            response: BulkStatesResponse = client.get_bulk_state(
                store_name=self.store_name,
                keys=keys,
                parallelism=parallelism,
                states_metadata=states_metadata or {},
            )
            return response.items or []

    def save_state(
        self,
        key: str,
        value: Union[str, bytes],
        *,
        state_metadata: Optional[Dict[str, str]] = None,
        etag: Optional[str] = None,
        state_options: Optional[Union[StateOptions, Dict[str, Any]]] = None,
    ) -> None:
        """
        Save a single key with raw bytes/str value.

        Args:
            key: Key to write.
            value: Bytes or string payload (caller handles JSON if desired).
            state_metadata: Optional Dapr metadata.
            etag: Optional ETag for concurrency.
            state_options: `StateOptions` or dict fields for options.
        """
        options = _coerce_state_options(state_options)
        with DaprClient() as client:
            client.save_state(
                store_name=self.store_name,
                key=key,
                value=value,
                state_metadata=state_metadata,
                etag=etag,
                options=options,
            )

    def save_bulk_state(
        self,
        states: List[StateItem],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Save multiple `StateItem`s. Caller constructs `StateItem` objects.

        Args:
            states: List of StateItem to write.
            metadata: Optional request metadata.
        """
        with DaprClient() as client:
            client.save_bulk_state(
                store_name=self.store_name,
                states=states,
                metadata=metadata,
            )

    def delete_state(
        self,
        key: str,
        *,
        etag: Optional[str] = None,
        state_options: Optional[Union[StateOptions, Dict[str, Any]]] = None,
        state_metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Delete a single key.

        Args:
            key: Key to delete.
            etag: Optional ETag for concurrency.
            state_options: `StateOptions` or dict of options.
            state_metadata: Optional Dapr metadata.
        """
        options = _coerce_state_options(state_options)
        with DaprClient() as client:
            client.delete_state(
                store_name=self.store_name,
                key=key,
                etag=etag,
                options=options,
                state_metadata=state_metadata,
            )

    def query_state(
        self,
        query: str,
        *,
        states_metadata: Optional[Dict[str, str]] = None,
    ) -> QueryResponse:
        """
        Execute a state query (backend must support Dapr state queries).

        Args:
            query: JSON query string.
            states_metadata: Optional Dapr metadata.

        Returns:
            `QueryResponse` containing results and metadata.
        """
        with DaprClient() as client:
            return client.query_state(
                store_name=self.store_name,
                query=query,
                states_metadata=states_metadata,
            )

    def execute_state_transaction(
        self,
        operations: Sequence[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Execute a transactional batch of operations.

        Args:
            operations: Dapr transaction operations (upserts/deletes).
            metadata: Optional request metadata.

        Note:
            Backend must support transactions (e.g., Redis in certain modes).
        """
        with DaprClient() as client:
            client.execute_state_transaction(
                store_name=self.store_name,
                operations=list(operations),
                metadata=metadata,
            )
