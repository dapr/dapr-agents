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

from dapr_agents.document.reader.base import ReaderBase
from dapr_agents.types.document import Document
from pathlib import Path
from typing import List
from pydantic import Field


class TextLoader(ReaderBase):
    """
    Loader for plain text files.

    Attributes:
        encoding (str): The text file encoding. Defaults to 'utf-8'.
    """

    encoding: str = Field(default="utf-8", description="Encoding of the text file.")

    def load(self, file_path: Path) -> List[Document]:
        """
        Load content from a plain text file.

        Args:
            file_path (Path): Path to the text file.

        Returns:
            List[Document]: A list containing one Document object.
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = file_path.read_text(encoding=self.encoding).strip()
        metadata = {
            "file_path": str(file_path),
            "file_type": "text",
        }
        return [Document(text=content, metadata=metadata)]
