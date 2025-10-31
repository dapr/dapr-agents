from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# TODO: rm this when we have the agent config class merged!
class Context(BaseModel):
    name: str = Field(
        default="Dapr Agent",
        description="The agent's name, defaulting to the role if not provided.",
    )
    role: Optional[str] = Field(
        default="Assistant",
        description="The agent's role in the interaction (e.g., 'Weather Expert').",
    )
    goal: Optional[str] = Field(
        default="Help humans",
        description="The agent's main objective (e.g., 'Provide Weather information').",
    )
    # TODO: add a background/backstory field that would be useful for the agent to know about it's context/background for it's role.
    instructions: Optional[List[str]] = Field(
        default=None, description="Instructions guiding the agent's tasks."
    )
    date: Optional[str] = Field(
        default=datetime.now().strftime("%B %d, %Y"),
        description="Date to use for the prompt context",
    )
