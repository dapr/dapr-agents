from dapr.clients.grpc._response import (
    BulkStatesResponse,
    BulkStateItem,
    StateResponse,
    QueryResponse,
)
from dapr.clients import DaprClient
from dapr.clients.grpc._state import StateItem
from dapr_agents.storage.daprstores.base import DaprStoreBase
from typing import Optional, Union, Dict, List, Tuple


class DaprStateStore(DaprStoreBase):
    def get_state(
        self,
        key: str,
        state_metadata: Optional[Dict[str, str]] = dict(),
    ) -> StateResponse:
        """
        Retrieves a value from the state store using the provided key.

        Args:
            key (str): The key for the state store item.
            state_metadata (Dict[str, str], optional): Dapr metadata for state request

        Returns:
            StateResponse: gRPC metadata returned from callee and value obtained from the state store
        """
        with DaprClient() as client:
            response: StateResponse = client.get_state(
                store_name=self.store_name, key=key, state_metadata=state_metadata
            )
            return response

    def try_get_state(
        self, key: str, state_metadata: Optional[Dict[str, str]] = dict()
    ) -> Tuple[bool, Optional[dict]]:
        """
        Attempts to retrieve a value from the state store using the provided key.

        Args:
            key (str): The key for the state store item.
            state_metadata (Dict[str, str], optional): Dapr metadata for state request.

        Returns:
            Tuple[bool, Optional[dict]]: A tuple where the first element is a boolean indicating whether the state exists,
                                        and the second element is the retrieved state data or None if not found.
        """
        with DaprClient() as client:
            response: StateResponse = client.get_state(
                store_name=self.store_name, key=key, state_metadata=state_metadata
            )
            if response and response.data:
                return True, response.json()
            return False, None

    def get_bulk_state(
        self,
        keys: List[str],
        parallelism: int = 1,
        states_metadata: Optional[Dict[str, str]] = None,
    ) -> List[BulkStateItem]:
        """
        Retrieves multiple values from the state store in bulk using a list of keys.

        Args:
            keys (List[str]): The keys to retrieve in bulk.
            parallelism (int, optional): Number of keys to retrieve in parallel.
            states_metadata (Dict[str, str], optional): Metadata for state request.

        Returns:
            List[BulkStateItem]: A list of BulkStateItem objects representing the retrieved state.
        """
        states_metadata = states_metadata or {}

        with DaprClient() as client:
            response: BulkStatesResponse = client.get_bulk_state(
                store_name=self.store_name,
                keys=keys,
                parallelism=parallelism,
                states_metadata=states_metadata,
            )

            if response and response.items:
                return response.items
            return []

    def save_state(
        self,
        key: str,
        value: Union[str, bytes],
        state_metadata: Optional[Dict[str, str]] = dict(),
    ):
        """
        Saves a key-value pair in the state store.

        Args:
            key (str): The key to save.
            value (Union[str, bytes]): The value to save.
            state_metadata (Dict[str, str], optional): Dapr metadata for state request
        """
        with DaprClient() as client:
            client.save_state(
                store_name=self.store_name,
                key=key,
                value=value,
                state_metadata=state_metadata,
            )

    def save_bulk_state(
        self, states: List[StateItem], metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Saves multiple key-value pairs to the state store in bulk.

        Args:
            states (List[StateItem]): The list of key-value pairs to save.
            metadata (Dict[str, str], optional): Metadata for the save request.
        """
        with DaprClient() as client:
            client.save_bulk_state(
                store_name=self.store_name, states=states, metadata=metadata
            )

    def delete_state(self, key: str):
        """
        Deletes a key-value pair from the state store.

        Args:
            key (str): The key to delete.
        """
        with DaprClient() as client:
            client.delete_state(store_name=self.store_name, key=key)

    def query_state(
        self, query: str, states_metadata: Optional[Dict[str, str]] = None
    ) -> QueryResponse:
        """
        Queries the state store with a specific query.

        Args:
            query (str): The query to be executed (in JSON format).
            states_metadata (Dict[str, str], optional): Custom metadata for the state request.

        Returns:
            QueryResponse: Contains query results and metadata.
        """
        with DaprClient() as client:
            client.query_state(
                store_name=self.store_name, query=query, states_metadata=states_metadata
            )
