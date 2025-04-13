from dapr_agents.document.embedder.base import EmbedderBase
from dapr_agents.llm.openai.embeddings import OpenAIEmbeddingClient
from typing import List, Any, Union, Optional
from pydantic import Field, ConfigDict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OpenAIEmbedder(OpenAIEmbeddingClient, EmbedderBase):
    """
    OpenAI-based embedder for generating text embeddings with handling for long inputs.
    Inherits functionality from OpenAIEmbeddingClient for API interactions.
    """

    max_tokens: int = Field(
        default=8191, description="Maximum tokens allowed per input."
    )
    chunk_size: int = Field(
        default=1000, description="Batch size for embedding requests."
    )
    normalize: bool = Field(
        default=True, description="Whether to normalize embeddings."
    )
    encoding_name: Optional[str] = Field(
        default=None, description="Token encoding name (if provided)."
    )
    encoder: Optional[Any] = Field(
        default=None, init=False, description="TikToken Encoder"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize attributes after model validation.
        Automatically determines the appropriate encoding for the model.
        """
        super().model_post_init(__context)

        try:
            import tiktoken
            from tiktoken.core import Encoding
        except ImportError:
            raise ImportError(
                "The `tiktoken` library is required for tokenizing inputs. "
                "Install it using `pip install tiktoken`."
            )

        if self.encoding_name:
            # Use the explicitly provided encoding
            self.encoder: Encoding = tiktoken.get_encoding(self.encoding_name)
        else:
            # Automatically determine encoding based on model
            try:
                self.encoder: Encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to default encoding and model
                logger.warning(
                    f"Model '{self.model}' not recognized. "
                    "Defaulting to 'cl100k_base' encoding and 'text-embedding-ada-002' model."
                )
                self.encoder = tiktoken.get_encoding("cl100k_base")
                self.model = "text-embedding-ada-002"

    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenizes the input text using the specified encoding."""
        return self.encoder.encode(text)

    def _chunk_tokens(self, tokens: List[int], chunk_length: int) -> List[List[int]]:
        """Splits tokens into chunks of the specified length."""
        return [
            tokens[i : i + chunk_length] for i in range(0, len(tokens), chunk_length)
        ]

    def _process_embeddings(
        self, embeddings: List[List[float]], weights: List[int]
    ) -> List[float]:
        """Combines embeddings using weighted averaging."""
        weighted_avg = np.average(embeddings, axis=0, weights=weights)
        if self.normalize:
            norm = np.linalg.norm(weighted_avg)
            return (weighted_avg / norm).tolist()
        return weighted_avg.tolist()

    def embed(
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Embeds input text(s) with support for both single and multiple inputs, handling long texts via chunking and batching.

        Args:
            input (Union[str, List[str]]): The input text(s) to embed. Can be a single string or a list of strings.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
                - Returns a single list of floats for a single string input.
                - Returns a list of lists of floats for a list of string inputs.

        Notes:
            - Handles long inputs by chunking them into smaller parts based on `max_tokens` and reassembling embeddings.
            - Batches API calls for efficiency using `chunk_size`.
            - Automatically combines chunk embeddings using weighted averaging for long inputs.
        """
        # Validate input
        if not input or (isinstance(input, list) and all(not q for q in input)):
            raise ValueError("Input must contain valid text.")

        # Check if the input is a single string or a list of strings
        single_input = isinstance(input, str)
        input_strings = [input] if single_input else input

        # Tokenize the input strings to check for long texts requiring chunking
        tokenized_inputs = [self._tokenize_text(q) for q in input_strings]
        chunks = []  # Holds text chunks for API calls
        chunk_indices = []  # Maps each chunk to its original input index

        # Handle tokenized inputs: Chunk long inputs and map chunks to their respective inputs
        for idx, tokens in enumerate(tokenized_inputs):
            if len(tokens) <= self.max_tokens:
                # Directly use the text if it's within max token limits
                chunks.append(self.encoder.decode(tokens))
                chunk_indices.append(idx)
            else:
                # Split long inputs into smaller chunks
                token_chunks = self._chunk_tokens(tokens, self.max_tokens)
                chunks.extend([self.encoder.decode(chunk) for chunk in token_chunks])
                chunk_indices.extend([idx] * len(token_chunks))

        # Process the chunks in batches for efficiency
        batch_size = self.chunk_size
        chunk_embeddings = []  # Holds embeddings for all chunks

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            response = self.create_embedding(input=batch)  # Batch API call
            chunk_embeddings.extend(r.embedding for r in response.data)

        # Group chunk embeddings by their original query indices
        grouped_embeddings = [[] for _ in range(len(input_strings))]
        for idx, embedding in zip(chunk_indices, chunk_embeddings):
            grouped_embeddings[idx].append(embedding)

        # Combine chunk embeddings for each query
        results = []
        for embeddings, tokens in zip(grouped_embeddings, tokenized_inputs):
            if len(embeddings) == 1:
                # If only one chunk, use its embedding directly
                results.append(embeddings[0])
            else:
                # Combine chunk embeddings using weighted averaging
                weights = [
                    len(chunk) for chunk in self._chunk_tokens(tokens, self.max_tokens)
                ]
                results.append(self._process_embeddings(embeddings, weights))

        # Return a single embedding if the input was a single string; otherwise, return a list
        return results[0] if single_input else results

    def __call__(
        self, input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Allows the instance to be called directly to embed text(s).

        Args:
            input (Union[str, List[str]]): The input text(s) to embed.

        Returns:
            Union[List[float], List[List[float]]]: Embedding vector(s) for the input(s).
        """
        return self.embed(input)
