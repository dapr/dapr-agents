import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from dapr_agents.memory import MemoryBase
from dapr_agents.storage.vectorstores import VectorStoreBase
from dapr_agents.types import AssistantMessage, MessageContent, UserMessage

logger = logging.getLogger(__name__)


class ConversationVectorMemory(MemoryBase):
    """
    Memory storage using a vector store, managing data storage and retrieval in a vector store for conversation sessions.
    """

    vector_store: VectorStoreBase = Field(
        ..., description="The vector store instance used for message storage."
    )

    def add_message(
        self,
        message: Union[Dict[str, Any], MessageContent],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds a single message to the vector store.

        Args:
            message: The message to add to the vector store.
            workflow_instance_id: Workflow instance id for this message.
        """
        message_dict = self._convert_to_dict(message)
        metadata = {
            "role": message_dict.get("role"),
            f"{message_dict.get('role')}_message": message_dict.get("content"),
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_instance_id": workflow_instance_id,
        }
        self.vector_store.add(
            documents=[message_dict.get("content")], metadatas=[metadata]
        )

    def add_messages(
        self,
        messages: List[Union[Dict[str, Any], MessageContent]],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds multiple messages to the vector store.

        Args:
            messages: A list of messages to add to the vector store.
            workflow_instance_id: Workflow instance id for these messages.
        """
        contents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for msg in messages:
            msg_dict = self._convert_to_dict(msg)
            contents.append(msg_dict.get("content"))
            metadatas.append(
                {
                    "role": msg_dict.get("role"),
                    f"{msg_dict.get('role')}_message": msg_dict.get("content"),
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "workflow_instance_id": workflow_instance_id,
                }
            )
        self.vector_store.add(contents, metadatas)

    def add_interaction(
        self,
        user_message: Union[Dict[str, Any], UserMessage],
        assistant_message: Union[Dict[str, Any], AssistantMessage],
        workflow_instance_id: str,
    ) -> None:
        """
        Adds a user-assistant interaction to the vector store as a single document.

        Args:
            user_message: The user message.
            assistant_message: The assistant message.
            workflow_instance_id: Workflow instance id for this interaction.
        """
        user_msg_dict = self._convert_to_dict(user_message)
        assistant_msg_dict = self._convert_to_dict(assistant_message)
        conversation_id = str(uuid.uuid4())
        conversation_text = f"User: {user_msg_dict.get('content')}\nAssistant: {assistant_msg_dict.get('content')}"
        conversation_embeddings = self.vector_store.embed_documents([conversation_text])
        metadata = {
            "user_message": user_msg_dict.get("content"),
            "assistant_message": assistant_msg_dict.get("content"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_instance_id": workflow_instance_id,
        }
        self.vector_store.add(
            documents=[conversation_text],
            embeddings=conversation_embeddings,
            metadatas=[metadata],
            ids=[conversation_id],
        )

    def get_messages(
        self,
        workflow_instance_id: str,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Retrieves messages from the vector store. If a query is provided, it performs a similarity search.

        Args:
            workflow_instance_id: Workflow instance id to retrieve messages for.
            query_embeddings: The query embeddings for similarity search. Defaults to None.
            k: The number of similar results to retrieve. Defaults to 4.
            distance_metric: The distance metric to use ("l2", "ip", "cosine"). Defaults to "cosine".

        Returns:
            A list of all stored or similar messages as dictionaries with 'role' and 'content'.
        """
        if query_embeddings:
            logger.info("Getting conversations related to user's query...")
            return self.get_similar_conversation(
                query_embeddings=query_embeddings, k=k, distance_metric=distance_metric
            )

        logger.info("Getting all conversations.")
        items = self.vector_store.get(include=["documents", "metadatas"])
        messages: List[Dict[str, Any]] = []
        for item in items:
            metadata = item["metadata"]
            if (
                metadata
                and "user_message" in metadata
                and "assistant_message" in metadata
            ):
                messages.append({"role": "user", "content": metadata["user_message"]})
                messages.append(
                    {"role": "assistant", "content": metadata["assistant_message"]}
                )
        return messages

    def reset_memory(self, workflow_instance_id: str) -> None:
        """
        Clears all messages from the vector store for the given workflow instance.

        Args:
            workflow_instance_id: Workflow instance id to reset.
        """
        self.vector_store.reset()

    def get_similar_conversation(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        k: int = 4,
        distance_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Performs a similarity search in the vector store and retrieves the conversation pairs.

        Args:
            query_embeddings (Optional[List[List[float]]], optional): The query embeddings. Defaults to None.
            k (int, optional): The number of results to return. Defaults to 4.
            distance_metric (str, optional): The distance metric to use ("l2", "ip", "cosine"). Defaults to "cosine".

        Returns:
            List[Dict[str, Any]]: A list of user and assistant messages in chronological order, each as a dictionary with 'role', 'content', and 'timestamp'.
        """
        distance_thresholds = {"l2": 1.0, "ip": 0.5, "cosine": 0.75}
        distance_threshold = distance_thresholds.get(distance_metric, 0.75)
        results = self.vector_store.search_similar(
            query_embeddings=query_embeddings, k=k
        )
        messages: List[Dict[str, Any]] = []

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
                        messages.append(
                            {
                                "role": "user",
                                "content": metadata["user_message"],
                                "timestamp": timestamp,
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": metadata["assistant_message"],
                                "timestamp": timestamp,
                            }
                        )
                    elif "user_message" in metadata:
                        messages.append(
                            {
                                "role": "user",
                                "content": metadata["user_message"],
                                "timestamp": timestamp,
                            }
                        )
                    elif "assistant_message" in metadata:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": metadata["assistant_message"],
                                "timestamp": timestamp,
                            }
                        )

        messages.sort(key=lambda x: x.get("timestamp"))
        return messages
