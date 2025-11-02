from __future__ import annotations

from pydantic import BaseModel, Field


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")
