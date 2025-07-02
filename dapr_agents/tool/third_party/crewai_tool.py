from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator, Field

from dapr_agents.tool import AgentTool
from dapr_agents.tool.utils.function_calling import to_function_call_definition
from dapr_agents.types import ToolError

# Try to import CrewAI BaseTool to support proper type checking
try:
    from crewai.tools import BaseTool as CrewAIBaseTool
except ImportError:
    # Define a placeholder for static type checking
    class CrewAIBaseTool:
        pass


class CrewAITool(AgentTool):
    """
    Adapter for using CrewAI tools with dapr-agents.

    This class wraps a CrewAI tool and makes it compatible with the dapr-agents
    framework, preserving the original tool's name, description, and schema.
    """

    tool: Any = Field(default=None)
    """The wrapped CrewAI tool."""

    def __init__(self, tool: Any, **kwargs):
        """
        Initialize the CrewAITool with a CrewAI tool.

        Args:
            tool: The CrewAI tool to wrap
            **kwargs: Optional overrides for name, description, etc.
        """
        # Extract metadata from CrewAI tool
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")

        # If name/description not provided via kwargs, extract from tool
        if not name:
            # Get name from the tool and format it (CrewAI tools often have spaces)
            raw_name = getattr(tool, "name", tool.__class__.__name__)
            name = raw_name.replace(" ", "_").title()

        if not description:
            # Get description from the tool
            description = getattr(tool, "description", tool.__doc__ or "")

        # Initialize the AgentTool with the CrewAI tool's metadata
        super().__init__(name=name, description=description)

        # Set the tool after parent initialization
        self.tool = tool

    @model_validator(mode="before")
    @classmethod
    def populate_name(cls, data: Any) -> Any:
        # Override name validation to properly format CrewAI tool name
        return data

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the wrapped CrewAI tool.

        Attempts to call the tool's run method or _execute method, depending on what's available.

        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            str: The result of the tool execution

        Raises:
            ToolError: If the tool execution fails
        """
        try:
            # Try different calling patterns based on CrewAI tool implementation
            if hasattr(self.tool, "run"):
                return self.tool.run(*args, **kwargs)
            elif hasattr(self.tool, "_execute"):
                return self.tool._execute(*args, **kwargs)
            elif callable(self.tool):
                return self.tool(*args, **kwargs)
            else:
                raise ToolError(f"Cannot execute CrewAI tool: {self.tool}")
        except Exception as e:
            raise ToolError(f"Error executing CrewAI tool: {str(e)}")

    def model_post_init(self, __context: Any) -> None:
        """Initialize args_model from the CrewAI tool schema if available."""
        super().model_post_init(__context)

        # Try to use the CrewAI tool's schema if available
        if hasattr(self.tool, "args_schema"):
            self.args_model = self.tool.args_schema

    def to_function_call(
        self, format_type: str = "openai", use_deprecated: bool = False
    ) -> Dict:
        """
        Converts the tool to a function call definition based on its schema.

        If the CrewAI tool has an args_schema, use it directly.

        Args:
            format_type (str): The format type (e.g., 'openai').
            use_deprecated (bool): Whether to use deprecated format.

        Returns:
            Dict: The function call representation.
        """
        # Use to_function_call_definition from function_calling utility
        if hasattr(self.tool, "args_schema") and self.tool.args_schema:
            # For CrewAI tools, we have their schema model directly
            return to_function_call_definition(
                self.name,
                self.description,
                self.args_model,
                format_type,
                use_deprecated,
            )
        else:
            # Fallback to the regular AgentTool implementation
            return super().to_function_call(format_type, use_deprecated)
