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

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class Document(BaseModel):
    """
    Represents a document with text content and associated metadata.
    """

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary containing metadata about the document (e.g., source, page number).",
    )
    text: str = Field(..., description="The main content of the document.")
