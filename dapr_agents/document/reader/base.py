#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
