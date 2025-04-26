from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.types import MessageContent, UserMessage, AssistantMessage
from dapr_agents.memory import MemoryBase
from datetime import datetime, timezone
from pydantic import Field
from typing import List, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


class ConversationVectorMemory(MemoryBase):
    """
    Memory storage using a vector store, managing data storage and retrieval in a vector store for conversation sessions.
    """

    vector_store: VectorStoreBase = Field(
        ..., description="The vector store instance used for message storage."
    )

    def add_message(self, message: MessageContent):
        """
        Adds a single message to the vector store.

        Args:
            message (MessageContent): The message to add to the vector store.
        """
        metadata = {
            "role": message.role,
            f"{message.role}_message": message.content,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.vector_store.add(documents=[message.content], metadatas=[metadata])

    def add_messages(self, messages: List[MessageContent]):
        """
        Adds multiple messages to the vector store.

        Args:
            messages (List[MessageContent]): A list of messages to add to the vector store.
        """
        contents = [msg.content for msg in messages]
        metadatas = [
            {
                "role": msg.role,
                f"{msg.role}_message": msg.content,
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for msg in messages
        ]
        self.vector_store.add(contents, metadatas)

    def add_interaction(
        self, user_message: UserMessage, assistant_message: AssistantMessage
    ):
        """
        Adds a user-assistant interaction to the vector store as a single document.

        Args:
            user_message (UserMessage): The user message.
            assistant_message (AssistantMessage): The assistant message.
        """
        conversation_id = str(uuid.uuid4())
        conversation_text = (
            f"User: {user_message.content}\nAssistant: {assistant_message.content}"
        )
        conversation_embeddings = self.vector_store.embed_documents([conversation_text])
        metadata = {
            "user_message": user_message.content,
            "assistant_message": assistant_message.content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.vector_store.add(
            documents=[conversation_text],
            embeddings=conversation_embeddings,
            metadatas=[metadata],
            ids=[conversation_id],
        )

    def get_messages(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        distance_metric: str = "cosine",
    ) -> List[MessageContent]:
        """
        Retrieves messages from the vector store. If a query is provided, it performs a similarity search.

        Args:
            query_embeddings (Optional[List[List[float]]]): The query embeddings for similarity search.
            k (int): The number of similar results to retrieve.
            distance_metric (str): The distance metric to use ("l2", "ip", "cosine").

        Returns:
            List[MessageContent]: A list of all stored or similar messages.
        """
        if query_embeddings:
            logger.info("Getting conversations related to user's query...")
            return self.get_similar_conversation(
                query_embeddings=query_embeddings, k=k, distance_metric=distance_metric
            )

        logger.info("Getting all conversations.")
        items = self.vector_store.get(include=["documents", "metadatas"])
        messages = []
        for item in items:
            metadata = item["metadata"]
            if (
                metadata
                and "user_message" in metadata
                and "assistant_message" in metadata
            ):
                messages.append(UserMessage(metadata["user_message"]))
                messages.append(AssistantMessage(metadata["assistant_message"]))
        return messages

    def reset_memory(self):
        """Clears all messages from the vector store."""
        self.vector_store.reset()

    def get_similar_conversation(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        distance_metric: str = "cosine",
    ) -> List[MessageContent]:
        """
        Performs a similarity search in the vector store and retrieves the conversation pairs.

        Args:
            query_embeddings (Optional[List[List[float]]]): The query embeddings.
            k (int): The number of results to return.
            distance_metric (str): The distance metric to use ("l2", "ip", "cosine").

        Returns:
            List[MessageContent]: A list of user and assistant messages in chronological order.
        """
        distance_thresholds = {"l2": 1.0, "ip": 0.5, "cosine": 0.75}
        distance_threshold = distance_thresholds.get(distance_metric, 0.75)
        results = self.vector_store.search_similar(
            query_embeddings=query_embeddings, k=k
        )
        messages = []

        if not results or not results["ids"][0]:
            return (
                messages  # Return an empty list if no similar conversations are found
            )

        for idx, distance in enumerate(results["distances"][0]):
            if distance <= distance_threshold:
                metadata = results["metadatas"][0][idx]
                if metadata:
                    timestamp = metadata.get("timestamp")
                    if "user_message" in metadata and "assistant_message" in metadata:
                        user_message = UserMessage(metadata["user_message"])
                        assistant_message = AssistantMessage(
                            metadata["assistant_message"]
                        )
                        messages.append((user_message, assistant_message, timestamp))
                    elif "user_message" in metadata:
                        user_message = UserMessage(metadata["user_message"])
                        messages.append((user_message, None, timestamp))
                    elif "assistant_message" in metadata:
                        assistant_message = AssistantMessage(
                            metadata["assistant_message"]
                        )
                        messages.append((None, assistant_message, timestamp))

        messages.sort(key=lambda x: x[2])
        sorted_messages = [msg for pair in messages for msg in pair[:2] if msg]

        return sorted_messages
