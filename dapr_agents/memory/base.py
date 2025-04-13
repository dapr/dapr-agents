from dapr_agents.types import BaseMessage
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
from typing import List


class MemoryBase(BaseModel, ABC):
    """
    Abstract base class for managing message memory. This class defines a standard interface for memory operations,
    allowing for different implementations of message storage mechanisms in subclasses.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def add_message(self, message: BaseMessage):
        """
        Adds a single message to the memory storage.

        Args:
            message (BaseMessage): The message object to be added.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def add_messages(self, messages: List[BaseMessage]):
        """
        Adds a list of messages to the memory storage.

        Args:
            messages (List[BaseMessage]): A list of message objects to be added.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def add_interaction(
        self, user_message: BaseMessage, assistant_message: BaseMessage
    ):
        """
        Adds a user-assistant interaction to the memory storage.

        Args:
            user_message (BaseMessage): The user message.
            assistant_message (BaseMessage): The assistant message.
        """
        pass

    @abstractmethod
    def get_messages(self) -> List[BaseMessage]:
        """
        Retrieves all messages from the memory storage.

        Returns:
            List[BaseMessage]: A list of all stored messages.

        Note:
            This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def reset_memory(self):
        """
        Clears all messages from the memory storage.

        Note:
            This method must be implemented by subclasses.
        """
        pass
