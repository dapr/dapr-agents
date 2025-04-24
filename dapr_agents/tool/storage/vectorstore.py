from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from dapr_agents.storage.vectorstores import VectorStoreBase
import logging

logger = logging.getLogger(__name__)


class VectorToolStore(BaseModel):
    """
    Manages tool information within a vector store, providing methods for adding tools and
    retrieving similar tools based on queries.
    """

    vector_store: VectorStoreBase = Field(
        ..., description="The vector store instance for tool data storage."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_tools(self, tools: List[Dict[str, Any]]):
        """
        Adds tool information to the vector store.

        Args:
            tools (List[Dict[str, Any]]): A list of dictionaries representing tools, each containing
                definitions and optional metadata for each tool.
        """
        logger.info("Adding tools to Vector Tool Store.")

        documents = []
        metadatas = []

        for tool in tools:
            func_name = tool["definition"]["function"]["name"]
            description = tool["definition"]["function"]["description"]
            parameters = tool["definition"]["function"]["parameters"]

            # Prepare the document for each tool
            documents.append(f"{func_name}: {description}. Args schema: {parameters}")

            # Prepare metadata, ensuring 'name' is always set
            metadata = tool.get("metadata", {}).copy()
            metadata.setdefault("name", func_name)  # Ensure name is set in metadata
            metadatas.append(metadata)

        self.vector_store.add(documents=documents, metadatas=metadatas)

    def get_similar_tools(self, query_texts: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves tools from the vector store similar to the query text.

        Args:
            query_texts (str): The query string to find similar tools.
            k (int): The number of similar results to return. Defaults to 4.

        Returns:
            List[Dict[str, Any]]: List of similar tool entries based on the query.
        """
        logger.info(f"Searching for tools similar to query: {query_texts}")
        similar_docs = self.vector_store.search_similar(query_texts=query_texts, k=k)
        return similar_docs

    def get_tool_names(self) -> List[str]:
        """
        Retrieves the names of all tools stored in the vector store.

        Returns:
            List[str]: A list of tool names.
        """
        logger.info("Retrieving all tool names from Vector Tool Store.")
        tools = self.vector_store.get()
        return [tool["metadata"]["name"] for tool in tools]
