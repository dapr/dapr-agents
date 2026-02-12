from dapr_agents.memory import MemoryBase
from dapr_agents.types import BaseMessage, MessageContent
from typing import List
from unittest.mock import Mock
from pydantic import PrivateAttr


class DummyVectorMemory(MemoryBase):
    """Mock vector memory for testing."""

    _vector_store = PrivateAttr()

    def __init__(self, vector_store):
        super().__init__()
        self._vector_store = vector_store

    def get_messages(self, workflow_instance_id: str, query_embeddings=None):
        return [Mock(spec=MessageContent)]

    def add_message(self, message: BaseMessage, workflow_instance_id: str):
        pass

    def add_messages(self, messages: List[BaseMessage], workflow_instance_id: str):
        pass

    def add_interaction(
        self,
        user_message: BaseMessage,
        assistant_message: BaseMessage,
        workflow_instance_id: str,
    ):
        pass

    def reset_memory(self, workflow_instance_id: str):
        pass
