from dapr_agents.storage.daprstores.base import DaprStoreBase
from typing import Dict, Optional


class DaprSecretStore(DaprStoreBase):
    def get_secret(
        self, key: str, secret_metadata: Optional[Dict[str, str]] = {}
    ) -> Optional[Dict[str, str]]:
        """
        Retrieves a secret from the secret store using the provided key.

        Args:
            key (str): The key for the secret.
            secret_metadata (Dict[str, str], optional): Metadata for the secret request.

        Returns:
            Optional[Dict[str, str]]: The secret stored in the secret store, or None if not found.
        """
        response = self.client.get_secret(
            store_name=self.store_name, key=key, secret_metadata=secret_metadata
        )
        return response.secret

    def get_bulk_secret(
        self, secret_metadata: Optional[Dict[str, str]] = {}
    ) -> Dict[str, Dict[str, str]]:
        """
        Retrieves all granted secrets from the secret store.

        Args:
            secret_metadata (Dict[str, str], optional): Metadata for the secret request.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary of secrets.
        """
        response = self.client.get_bulk_secret(
            store_name=self.store_name, secret_metadata=secret_metadata
        )
        return response.secrets
