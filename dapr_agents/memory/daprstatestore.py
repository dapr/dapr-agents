import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, model_validator

from dapr_agents.memory import MemoryBase
from dapr_agents.storage.daprstores.statestore import DaprStateStore
from dapr_agents.types import BaseMessage

logger = logging.getLogger(__name__)


def generate_numeric_session_id() -> int:
    """
    Generates a random numeric session ID by extracting digits from a UUID.

    Returns:
        int: A numeric session ID.
    """
    return int("".join(filter(str.isdigit, str(uuid.uuid4()))))


class ConversationDaprStateMemory(MemoryBase):
    """
    Manages conversation memory stored in a Dapr state store. Each message in the conversation is saved
    individually with a unique key and includes a session ID and timestamp for querying and retrieval.
    """

    store_name: str = Field(
        default="statestore", description="The name of the Dapr state store."
    )
    session_id: Optional[Union[str, int]] = Field(
        default=None, description="Unique identifier for the conversation session."
    )

    dapr_store: Optional[DaprStateStore] = Field(
        default=None, init=False, description="Dapr State Store."
    )

    @model_validator(mode="before")
    def set_session_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets a numeric session ID if none is provided.

        Args:
            values (Dict[str, Any]): The dictionary of attribute values before initialization.

        Returns:
            Dict[str, Any]: Updated values including the generated session ID if not provided.
        """
        if not values.get("session_id"):
            values["session_id"] = generate_numeric_session_id()
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes the Dapr state store after validation.
        """
        self.dapr_store = DaprStateStore(store_name=self.store_name)
        logger.info(
            f"ConversationDaprStateMemory initialized with session ID: {self.session_id}"
        )
        super().model_post_init(__context)

    def _get_message_key(self, message_id: str) -> str:
        """
        Generates a unique key for each message using session_id and message_id.

        Args:
            message_id (str): A unique identifier for the message.

        Returns:
            str: A composite key for storing individual messages.
        """
        return f"{self.session_id}:{message_id}"

    def add_message(self, message: Union[Dict[str, Any], BaseMessage]) -> None:
        """
        Adds a single message to the memory and saves it to the Dapr state store.

        Args:
            message (Union[Dict[str, Any], BaseMessage]): The message to add to the memory.
        """
        message = self._convert_to_dict(message)
        message_id = str(uuid.uuid4())
        message_key = self._get_message_key(message_id)
        message.update(
            {
                "sessionId": self.session_id,
                "createdAt": datetime.now().isoformat() + "Z",
            }
        )
        existing = self.get_messages()
        existing.append(message)
        logger.debug(
            f"Adding message {message} with key {message_key} to session {self.session_id}"
        )
        self.dapr_store.save_state(
            self.session_id, json.dumps(existing), {"contentType": "application/json"}
        )

    def add_messages(self, messages: List[Union[Dict[str, Any], BaseMessage]]) -> None:
        """
        Adds multiple messages to the memory and saves each one individually to the Dapr state store.

        Args:
            messages (List[Union[Dict[str, Any], BaseMessage]]): A list of messages to add to the memory.
        """
        logger.info(f"Adding {len(messages)} messages to session {self.session_id}")
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
        Retrieves messages stored in the state store for the current session_id, with an optional limit.

        Args:
            limit (int, optional): The maximum number of messages to retrieve. Defaults to 100.

        Returns:
            List[Dict[str, Any]]: A list of message dicts with all fields.
        """
        response = self.query_messages(session_id=self.session_id)
        if response and hasattr(response, "data") and response.data:
            raw_messages = json.loads(response.data)
            if raw_messages:
                messages = raw_messages[:limit]
                logger.info(
                    f"Retrieved {len(messages)} messages for session {self.session_id}"
                )
                return messages
        return []

    def query_messages(self, session_id: str) -> Any:
        """
        Queries messages from the state store for the given session_id.

        Args:
            session_id (str): The session ID to query messages for.

        Returns:
            Any: The response object from the Dapr state store, typically with a 'data' attribute containing the messages as JSON.
        """
        logger.debug(f"Executing query for session {self.session_id}")
        states_metadata = {"contentType": "application/json"}
        response = self.dapr_store.get_state(session_id, state_metadata=states_metadata)
        return response

    def reset_memory(self) -> None:
        """
        Clears all messages stored in the memory and resets the state store for the current session.
        """
        self.dapr_store.delete_state(self.session_id)
        logger.info(f"Memory reset for session {self.session_id} completed.")
