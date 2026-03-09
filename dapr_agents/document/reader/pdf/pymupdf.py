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
from typing import List, Dict, Optional
from pathlib import Path


class PyMuPDFReader(ReaderBase):
    """
    Reader for PDF documents using PyMuPDF.
    """

    def load(
        self, file_path: Path, additional_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load content from a PDF file using PyMuPDF.

        Args:
            file_path (Path): Path to the PDF file.
            additional_metadata (Optional[Dict]): Additional metadata to include.

        Returns:
            List[Document]: A list of Document objects.
        """
        try:
            import pymupdf
        except ImportError:
            raise ImportError(
                "PyMuPDF library is not installed. Install it using `pip install pymupdf`."
            )

        file_path = str(file_path)
        doc = pymupdf.open(file_path)
        total_pages = len(doc)
        documents = []

        for page_num, page in enumerate(doc.pages):
            text = page.get_text()
            metadata = {
                "file_path": file_path,
                "page_number": page_num + 1,
                "total_pages": total_pages,
            }
            if additional_metadata:
                metadata.update(additional_metadata)

            documents.append(Document(text=text.strip(), metadata=metadata))

        doc.close()
        return documents
