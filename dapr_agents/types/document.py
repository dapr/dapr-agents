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
