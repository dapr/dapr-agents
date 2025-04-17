from pydantic import BaseModel, ConfigDict, Field
from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from dapr_agents.types.document import Document
import re
import logging

try:
    from nltk.tokenize import sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    sent_tokenize = None
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class SplitterBase(BaseModel, ABC):
    """
    Base class for defining text splitting strategies.
    Provides common utilities for breaking text into smaller chunks
    based on separators, regex patterns, or sentence-based splitting.
    """

    chunk_size: int = Field(
        default=4000,
        description="Maximum size of chunks (in characters or tokens).",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap size between chunks for context continuity.",
        ge=0,
    )
    chunk_size_function: Callable[[str], int] = Field(
        default=len,
        description="Function to calculate chunk size (e.g., by characters or tokens).",
    )
    separator: Optional[str] = Field(
        default="\n\n", description="Primary separator for splitting text."
    )
    fallback_separators: List[str] = Field(
        default_factory=lambda: ["\n", " "],
        description="Fallback separators if the primary separator fails.",
    )
    fallback_regex: str = Field(
        default=r"[^,.;。？！]+[,.;。？！]",
        description="Improved regex pattern for fallback splitting.",
    )
    reserved_metadata_size: int = Field(
        default=0, description="Tokens reserved for metadata.", ge=0
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Abstract method to be implemented by child classes for splitting text.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: List of text chunks.
        """
        pass

    def _get_chunk_size(self, text: str) -> int:
        """
        Calculate the size of a chunk based on the provided chunk_size_function.

        Args:
            text (str): The input text.

        Returns:
            int: The size of the text chunk.
        """
        return self.chunk_size_function(text)

    def _merge_splits(self, splits: List[str], max_size: int) -> List[str]:
        """
        Merge splits into chunks while ensuring size constraints and meaningful overlaps.

        Unlike other implementations, this method prioritizes sentence boundaries
        when creating overlaps, ensuring that each chunk remains contextually meaningful.

        Args:
            splits (List[str]): The text segments to be merged.
            max_size (int): Maximum allowed size for each chunk.

        Returns:
            List[str]: The resulting merged chunks.
        """
        if not splits:
            return []

        chunks = []  # Store finalized chunks
        current_chunk = []  # Collect splits for the current chunk
        current_size = 0  # Track the size of the current chunk

        for split in splits:
            split_size = self._get_chunk_size(split)

            # If adding the current split exceeds max_size, finalize the current chunk
            if current_size + split_size > max_size:
                if current_chunk:
                    # Finalize the current chunk
                    full_chunk = "".join(current_chunk)
                    chunks.append(full_chunk)

                    # Logging information for overlap and chunk size
                    logger.debug(
                        f"Chunk {len(chunks)} finalized. Size: {current_size}. Overlap size: {self.chunk_overlap}"
                    )

                    # Create an overlap using sentences from the current chunk
                    overlap = []
                    overlap_size = 0
                    for sentence in reversed(current_chunk):
                        sentence_size = self._get_chunk_size(sentence)
                        if overlap_size + sentence_size > self.chunk_overlap:
                            break
                        overlap.insert(0, sentence)
                        overlap_size += sentence_size

                    # Logging information for overlap content
                    logger.debug(f"Chunk {len(chunks)} overlap: {''.join(overlap)}")

                    # Start the new chunk with the overlap
                    current_chunk = overlap
                    current_size = overlap_size
                else:
                    # If a single split exceeds max_size, treat it as a standalone chunk
                    chunks.append(split)
                    current_chunk = []
                    current_size = 0
            else:
                # Add the current split to the ongoing chunk
                current_chunk.append(split)
                current_size += split_size

        # Finalize the last chunk
        if current_chunk:
            chunks.append("".join(current_chunk))
            logger.debug(f"Chunk {len(chunks)} finalized. Size: {current_size}.")

        return chunks

    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Split text using a prioritized list of separators while keeping separators in chunks.

        For each separator in the provided list, attempt to split the text. The separator
        is appended to each split except the last one to preserve structure.

        Args:
            text (str): The input text to split.
            separators (List[str]): List of separators in order of priority.

        Returns:
            List[str]: A list of non-empty splits with separators retained.
        """
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                result = []
                for i, part in enumerate(parts):
                    # Add separator to all splits except the last one
                    if i < len(parts) - 1:
                        result.append(part + separator)
                    else:
                        result.append(part)
                # Return non-empty chunks only
                return [chunk for chunk in result if chunk.strip()]
        return [text.strip()]

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK if available, or fallback to regex.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: List of sentences split from the text.
        """
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        return self._regex_split(text)

    def _regex_split(self, text: str) -> List[str]:
        """
        Split text using the fallback regex, retaining separators.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: List of text segments split using regex.
        """
        matches = re.findall(self.fallback_regex, text)
        return [match for match in matches if match.strip()]

    def _split_adaptively(self, text: str) -> List[str]:
        """
        Adaptively split text using separators, fallback methods, and regex.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: List of adaptively split text segments.
        """
        # Try primary separator first
        chunks = self._split_by_separators(text, [self.separator])

        # Use fallback separators if the primary separator fails
        if len(chunks) <= 1:
            chunks = self._split_by_separators(text, self.fallback_separators)

        # Finally, fallback to sentence-based or regex splitting
        if len(chunks) <= 1:
            chunks = self._split_by_sentences(text)

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks while retaining metadata.

        Args:
            documents (List[Document]): List of documents to be split.

        Returns:
            List[Document]: List of chunked documents with updated metadata.
        """
        chunked_documents = []
        for doc in documents:
            text_chunks = self.split(doc.text)

            previous_end = 0
            for chunk_num, chunk in enumerate(text_chunks):
                start_index = doc.text.find(chunk, previous_end)
                if start_index == -1:
                    start_index = previous_end
                end_index = start_index + self._get_chunk_size(chunk)

                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata.update(
                    {
                        "chunk_number": chunk_num + 1,
                        "total_chunks": len(text_chunks),
                        "start_index": start_index,
                        "end_index": end_index,
                        "chunk_length": self._get_chunk_size(chunk),
                    }
                )
                chunked_documents.append(Document(metadata=metadata, text=chunk))
                previous_end = end_index

        return chunked_documents
