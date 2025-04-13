from dapr_agents.memory import MemoryBase
from dapr_agents.types import BaseMessage
from pydantic import Field
from typing import List, Dict, Union


class ConversationListMemory(MemoryBase):
    """
    Memory storage for conversation messages using a list-based approach. This class provides a simple way to store,
    retrieve, and manage messages during a conversation session.
    """

    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="List of messages stored in conversation memory.",
    )

    def add_message(self, message: Union[Dict, BaseMessage]):
        """
        Adds a single message to the end of the memory list.

        Args:
            message (Union[Dict, BaseMessage]): The message to add to the memory.
        """
        self.messages.append(self._convert_to_dict(message))

    def add_messages(self, messages: List[Union[Dict, BaseMessage]]):
        """
        Adds multiple messages to the memory by appending each message from the provided list to the end of the memory list.

        Args:
            messages (List[Union[Dict, BaseMessage]]): A list of messages to add to the memory.
        """
        self.messages.extend(self._convert_to_dict(msg) for msg in messages)

    def add_interaction(
        self, user_message: BaseMessage, assistant_message: BaseMessage
    ):
        """
        Adds a user-assistant interaction to the memory storage.

        Args:
            user_message (BaseMessage): The user message.
            assistant_message (BaseMessage): The assistant message.
        """
        self.add_messages([user_message, assistant_message])

    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieves a copy of all messages stored in the memory.

        Returns:
            List[BaseMessage]: A list containing copies of all stored messages.
        """
        return self.messages.copy()

    def reset_memory(self):
        """Clears all messages stored in the memory, resetting the memory to an empty state."""
        self.messages.clear()

    @staticmethod
    def _convert_to_dict(message: Union[Dict, BaseMessage]) -> Dict:
        """
        Converts a BaseMessage to a dictionary if necessary.

        Args:
            message (Union[Dict, BaseMessage]): The message to potentially convert.

        Returns:
            Dict: The message as a dictionary.
        """
        return message.model_dump() if isinstance(message, BaseMessage) else message
