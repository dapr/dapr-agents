from dapr_agents.document.splitter.base import SplitterBase
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextSplitter(SplitterBase):
    """
    Concrete implementation of the SplitterBase class.
    Splits text using a primary separator, fallback strategies,
    and applies size limits and overlap logic.
    """

    def split(self, text: str) -> List[str]:
        """
        Splits input text into chunks using a hierarchical strategy.

        Steps:
        1. Adjusts the effective chunk size to account for metadata space.
        2. If the text is smaller than the adjusted chunk size, returns it as a single chunk.
        3. Splits text adaptively using primary and fallback methods.
        4. Merges smaller chunks into valid sizes while maintaining overlap.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: List of merged text chunks.
        """
        # Step 1: Adjust effective chunk size
        effective_chunk_size = self.chunk_size - self.reserved_metadata_size
        logger.debug(f"Effective chunk size: {effective_chunk_size}")

        # Step 2: Short-circuit for small texts
        if self._get_chunk_size(text) <= effective_chunk_size:
            logger.debug(
                "Text size is smaller than effective chunk size. Returning as a single chunk."
            )
            return [text]

        # Step 3: Use adaptive splitting strategy
        chunks = self._split_adaptively(text)
        logger.debug(f"Initial split into {len(chunks)} chunks.")

        # Step 4: Merge smaller chunks into valid sizes with overlap
        merged_chunks = self._merge_splits(chunks, effective_chunk_size)
        logger.debug(f"Merged into {len(merged_chunks)} chunks with overlap.")

        return merged_chunks
