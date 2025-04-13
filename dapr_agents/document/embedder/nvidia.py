from dapr_agents.llm.nvidia.embeddings import NVIDIAEmbeddingClient
from dapr_agents.document.embedder.base import EmbedderBase
from typing import List, Union
from pydantic import Field
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NVIDIAEmbedder(NVIDIAEmbeddingClient, EmbedderBase):
    """
    NVIDIA-based embedder for generating text embeddings with support for indexing (passage) and querying.
    Inherits functionality from NVIDIAEmbeddingClient for API interactions.

    Attributes:
        chunk_size (int): Batch size for embedding requests. Defaults to 1000.
        normalize (bool): Whether to normalize embeddings. Defaults to True.
    """

    chunk_size: int = Field(
        default=1000, description="Batch size for embedding requests."
    )
    normalize: bool = Field(
        default=True, description="Whether to normalize embeddings."
    )

    def embed(
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Embeds input text(s) for indexing with default input_type set to 'passage'.

        Args:
            input (Union[str, List[str]]): Input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
                - Returns a single list of floats for a single string input.
                - Returns a list of lists of floats for a list of string inputs.

        Raises:
            ValueError: If input is invalid or embedding generation fails.
        """
        return self._generate_embeddings(input, input_type="passage")

    def embed_query(
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Embeds input text(s) for querying with input_type set to 'query'.

        Args:
            input (Union[str, List[str]]): Input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
                - Returns a single list of floats for a single string input.
                - Returns a list of lists of floats for a list of string inputs.

        Raises:
            ValueError: If input is invalid or embedding generation fails.
        """
        return self._generate_embeddings(input, input_type="query")

    def _generate_embeddings(
        self, input: Union[str, List[str]], input_type: str
    ) -> Union[List[float], List[List[float]]]:
        """
        Helper function to generate embeddings for given input text(s) with specified input_type.

        Args:
            input (Union[str, List[str]]): Input text(s) to embed.
            input_type (str): The type of embedding operation ('query' or 'passage').

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
        """
        # Validate input
        if not input or (isinstance(input, list) and all(not q for q in input)):
            raise ValueError("Input must contain valid text.")

        single_input = isinstance(input, str)
        input_list = [input] if single_input else input

        # Process input in chunks for efficiency
        chunk_embeddings = []
        for i in range(0, len(input_list), self.chunk_size):
            batch = input_list[i : i + self.chunk_size]
            response = self.create_embedding(input=batch, input_type=input_type)
            chunk_embeddings.extend(r.embedding for r in response.data)

        # Normalize embeddings if required
        if self.normalize:
            normalized_embeddings = [
                (embedding / np.linalg.norm(embedding)).tolist()
                for embedding in chunk_embeddings
            ]
        else:
            normalized_embeddings = chunk_embeddings

        # Return a single embedding if the input was a single string; otherwise, return a list
        return normalized_embeddings[0] if single_input else normalized_embeddings

    def __call__(
        self, input: Union[str, List[str]], query: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        Allows the instance to be called directly to embed text(s).

        Args:
            input (Union[str, List[str]]): The input text(s) to embed.
            query (bool): If True, embeds for querying (input_type='query'). Otherwise, embeds for indexing (input_type='passage').

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
        """
        if query:
            return self.embed_query(input)
        return self.embed(input)
