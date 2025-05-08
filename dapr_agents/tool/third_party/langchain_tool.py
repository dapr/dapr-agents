import inspect
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator, Field

from dapr_agents.tool import AgentTool
from dapr_agents.tool.utils.function_calling import to_function_call_definition
from dapr_agents.types import ToolError

# Try to import LangChain BaseTool to support proper type checking
try:
    from langchain_core.tools import BaseTool as LangchainBaseTool
except ImportError:
    # Define a placeholder for static type checking
    class LangchainBaseTool:
        pass

class LangchainTool(AgentTool):
    """
    Adapter for using LangChain tools with dapr-agents.

    This class wraps a LangChain tool and makes it compatible with the dapr-agents
    framework, preserving the original tool's name, description, and schema.
    """

    tool: Any = Field(default=None)
    """The wrapped LangChain tool."""

    def __init__(self, tool: Any, **kwargs):
        """
        Initialize the LangchainTool with a LangChain tool.

        Args:
            tool: The LangChain tool to wrap
            **kwargs: Optional overrides for name, description, etc.
        """
        # Extract metadata from LangChain tool
        name = kwargs.get("name", "")
        description = kwargs.get("description", "")
        
        # If name/description not provided via kwargs, extract from tool
        if not name:
            # Get name from the tool
            raw_name = getattr(tool, "name", tool.__class__.__name__)
            name = raw_name.replace(" ", "_").title()
            
        if not description:
            # Get description from the tool
            description = getattr(tool, "description", tool.__doc__ or "")
        
        # Initialize the AgentTool with the LangChain tool's metadata
        super().__init__(name=name, description=description)
        
        # Set the tool after parent initialization
        self.tool = tool
        
    @model_validator(mode="before")
    @classmethod
    def populate_name(cls, data: Any) -> Any:
        # Override name validation to properly format LangChain tool name
        return data

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the wrapped LangChain tool.
        
        Attempts to call the tool's _run method or run method, depending on what's available.
        
        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
            str: The result of the tool execution
            
        Raises:
            ToolError: If the tool execution fails
        """
        try:
            # Handle common issue where args/kwargs are passed differently
            # If 'args' is in kwargs, extract and use as the query
            if 'args' in kwargs and isinstance(kwargs['args'], list) and len(kwargs['args']) > 0:
                query = kwargs['args'][0]
                return self._run_with_query(query)
                
            # If args has content, use the first arg
            elif args and len(args) > 0:
                query = args[0]
                return self._run_with_query(query)
                
            # Otherwise, just pass through the kwargs
            else:
                return self._run_with_query(**kwargs)
        except Exception as e:
            raise ToolError(f"Error executing LangChain tool: {str(e)}")
    
    def _run_with_query(self, query=None, **kwargs):
        """Helper method to run the tool with different calling patterns."""
        try:
            # First check for single argument query-based pattern
            if query is not None:
                if hasattr(self.tool, "_run"):
                    return self.tool._run(query)
                elif hasattr(self.tool, "run"):
                    return self.tool.run(query)
                elif callable(self.tool):
                    return self.tool(query)
            
            # Fall back to kwargs pattern
            else:
                if hasattr(self.tool, "_run"):
                    return self.tool._run(**kwargs)
                elif hasattr(self.tool, "run"):
                    return self.tool.run(**kwargs)
                elif callable(self.tool):
                    return self.tool(**kwargs)
                    
            # If we get here, couldn't find a way to execute
            raise ToolError(f"Cannot execute LangChain tool: {self.tool}")
        except Exception as e:
            raise ToolError(f"Error executing LangChain tool: {str(e)}")
            
    def model_post_init(self, __context: Any) -> None:
        """Initialize args_model from the LangChain tool schema if available."""
        super().model_post_init(__context)
        
        # Try to use the LangChain tool's schema if available
        if hasattr(self.tool, "args_schema"):
            self.args_model = self.tool.args_schema
        elif hasattr(self.tool, "schema"):
            self.args_model = self.tool.schema
            
    def to_function_call(self, format_type: str = "openai", use_deprecated: bool = False) -> Dict:
        """
        Converts the tool to a function call definition based on its schema.
        
        If the LangChain tool has an args_schema, use it directly.
        
        Args:
            format_type (str): The format type (e.g., 'openai').
            use_deprecated (bool): Whether to use deprecated format.
            
        Returns:
            Dict: The function call representation.
        """
        # Use to_function_call_definition from function_calling utility
        if hasattr(self.tool, "args_schema") and self.tool.args_schema:
            # For LangChain tools, we have their schema model directly
            return to_function_call_definition(
                self.name, 
                self.description, 
                self.args_model, 
                format_type, 
                use_deprecated
            )
        else:
            # Fallback to the regular AgentTool implementation
            return super().to_function_call(format_type, use_deprecated) 