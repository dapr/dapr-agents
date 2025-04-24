from typing import List
from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Stores the outcome of a code execution."""

    status: str = Field(
        ..., description="The execution status, either 'success' or 'error'."
    )
    output: str = Field(
        ...,
        description="The standard output or error message resulting from execution.",
    )
    exit_code: int = Field(
        ...,
        description="The exit code returned by the executed process (0 indicates success, non-zero indicates an error).",
    )


class CodeSnippet(BaseModel):
    """Represents a block of code extracted for execution."""

    language: str = Field(
        ...,
        description="The programming language of the code snippet (e.g., 'python', 'javascript').",
    )
    code: str = Field(..., description="The actual source code to be executed.")
    timeout: int = Field(
        5,
        description="Per-snippet timeout (seconds). Executor falls back to the request-level timeout if omitted.",
    )


class ExecutionRequest(BaseModel):
    """Represents a request to execute a code snippet."""

    snippets: List[CodeSnippet] = Field(
        ...,
        description="A list of code snippets to be executed sequentially or in parallel.",
    )
    timeout: int = Field(
        5,
        description="The maximum time (in seconds) allowed for execution before timing out (default is 5 seconds).",
    )
