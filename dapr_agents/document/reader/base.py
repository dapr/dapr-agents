from dapr_agents.types.document import Document
from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
from typing import List


class ReaderBase(BaseModel, ABC):
    """
    Abstract base class for file readers.
    """

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """
        Load content from a file.

        Args:
            file_path (Path): Path to the file.

        Returns:
            List[Document]: A list of Document objects.
        """
        pass
