import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from dapr_agents.memory import MemoryBase
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.types import BaseMessage

logger = logging.getLogger(__name__)

class ConversationDaprStateMemory(MemoryBase):
    """
    Manages conversation memory stored in a Dapr state store. Each message in the conversation is saved
    individually with a unique key and includes a workflow instance ID and timestamp for querying and retrieval.
    """

    store_name: str = Field(
        default="statestore", description="The name of the Dapr state store."
    )
    workflow_instance_id: str = Field(
        default=None, description="Unique identifier for the conversation workflow instance summary."
    )

    dapr_store: Optional[DaprStateStore] = Field(
        default=None, init=False, description="Dapr State Store."
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the Dapr state store after validation.
        """
        self.dapr_store = DaprStateStore(store_name=self.store_name)
        logger.info(
            f"ConversationDaprStateMemory initialized with workflow instance ID: {self.workflow_instance_id}"
        )
        super().model_post_init(__context)

    def _get_message_key(self, message_id: str) -> str:
        """
        Generates a unique key for each message using workflow instance id and message_id.

        Args:
            message_id (str): A unique identifier for the message.

        Returns:
            str: A composite key for storing individual messages.
        """
        return f"{self.workflow_instance_id}:{message_id}"

    def add_message(self, message: Union[Dict[str, Any], BaseMessage]) -> None:
        """
        Adds a single message to the memory and saves it to the Dapr state store.

        Args:
            message (Union[Dict[str, Any], BaseMessage]): The message to add to the memory.
        """
        message = self._convert_to_dict(message)
        message.update(
            {
                "createdAt": datetime.now().isoformat() + "Z",
            }
        )

        # Retry loop for optimistic concurrency control
        # TODO: make this nicer in future, but for durability this must all be atomic
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.dapr_store.get_state(
                    self.workflow_instance_id,
                    state_metadata={"contentType": "application/json"},
                )

                if response and response.data:
                    existing = json.loads(response.data)
                    etag = response.etag
                else:
                    existing = []
                    etag = None

                existing.append(message)
                # Save with etag - will fail if someone else modified it
                self.dapr_store.save_state(
                    self.workflow_instance_id,
                    json.dumps(existing),
                    state_metadata={"contentType": "application/json"},
                    etag=etag,
                )

                # Success - exit retry loop
                return

            except Exception as exc:
                if attempt == max_attempts:
                    logger.exception(
                        f"Failed to add message to workflow instance {self.workflow_instance_id} after {max_attempts} attempts: {exc}"
                    )
                    raise
                else:
                    logger.warning(
                        f"Conflict adding message to workflow instance {self.workflow_instance_id} (attempt {attempt}/{max_attempts}): {exc}, retrying..."
                    )
                    # Brief exponential backoff with jitter
                    import time
                    import random

                    time.sleep(min(0.1 * attempt, 0.5) * (1 + random.uniform(0, 0.25)))

    def add_messages(self, messages: List[Union[Dict[str, Any], BaseMessage]]) -> None:
        """
        Adds multiple messages to the memory and saves each one individually to the Dapr state store.

        Args:
            messages (List[Union[Dict[str, Any], BaseMessage]]): A list of messages to add to the memory.
        """
        logger.info(f"Adding {len(messages)} messages to workflow instance {self.workflow_instance_id}")
        for message in messages:
            self.add_message(message)

    def add_interaction(
        self,
        user_message: Union[Dict[str, Any], BaseMessage],
        assistant_message: Union[Dict[str, Any], BaseMessage],
    ) -> None:
        """
        Adds a user-assistant interaction to the memory storage and saves it to the state store.

        Args:
            user_message (Union[Dict[str, Any], BaseMessage]): The user message.
            assistant_message (Union[Dict[str, Any], BaseMessage]): The assistant message.
        """
        self.add_messages([user_message, assistant_message])

    def _decode_message(self, message_data: Union[bytes, str]) -> Dict[str, Any]:
        """
        Decodes the message data if it's in bytes, otherwise parses it as a JSON string.

        Args:
            message_data (Union[bytes, str]): The message data to decode.

        Returns:
            Dict[str, Any]: The decoded message as a dictionary.
        """
        if isinstance(message_data, bytes):
            message_data = message_data.decode("utf-8")
        return json.loads(message_data)

    def get_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves messages stored in the state store for the current workflow_instance_id, with an optional limit.

        Args:
            limit (int, optional): The maximum number of messages to retrieve. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: A list of message dicts with all fields.
        """
        response = self.query_messages()
        if response and hasattr(response, "data") and response.data:
            raw_messages = json.loads(response.data)
            if raw_messages:
                messages = raw_messages[:limit]
                logger.info(
                    f"Retrieved {len(messages)} messages for workflow instance {self.workflow_instance_id}"
                )
                return messages
        return []

    def query_messages(self) -> Any:
        """
        Queries messages from the state store for the given workflow_instance_id.
        Returns:
            Any: The response object from the Dapr state store, typically with a 'data' attribute containing the messages as JSON.
        """
        logger.debug(f"Executing query for workflow instance {self.workflow_instance_id}")
        states_metadata = {"contentType": "application/json"}
        response = self.dapr_store.get_state(self.workflow_instance_id, state_metadata=states_metadata)
        return response

    def reset_memory(self) -> None:
        """
        Clears all messages stored in the memory and resets the state store for the current workflow instance.
        """
        self.dapr_store.delete_state(self.workflow_instance_id)
        logger.info(f"Memory reset for workflow instance {self.workflow_instance_id} completed.")