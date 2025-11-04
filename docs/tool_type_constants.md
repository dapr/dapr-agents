# Tool Type Constants

## Overview

Tool type constants provide a type-safe way to specify and check tool types without using magic strings throughout the codebase.

## Constants

```python
from dapr_agents.types import (
    TOOL_TYPE_FUNCTION,  # "function"
    TOOL_TYPE_MCP,       # "mcp"
    TOOL_TYPE_AGENT,     # "agent"
    TOOL_TYPE_UNKNOWN,   # "unknown"
    ToolType,            # Type alias: Literal["function", "mcp", "agent", "unknown"]
)
```

## Usage

### Creating Tool Definitions

```python
from dapr_agents.registry import ToolDefinition, TOOL_TYPE_FUNCTION, TOOL_TYPE_MCP

# Using constants instead of magic strings
function_tool = ToolDefinition(
    name="WeatherTool",
    description="Get weather information",
    tool_type=TOOL_TYPE_FUNCTION  # ✅ Type-safe
)

mcp_tool = ToolDefinition(
    name="DatabaseQuery",
    description="Query database",
    tool_type=TOOL_TYPE_MCP  # ✅ Type-safe
)

# Instead of:
# tool_type="function"  # ❌ Magic string, prone to typos
```

### Agent Tools

The `AgentTool` class automatically uses these constants:

```python
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import TOOL_TYPE_FUNCTION

def my_function():
    """My tool function"""
    pass

# AgentTool.from_func() automatically sets tool_type=TOOL_TYPE_FUNCTION
tool = AgentTool.from_func(my_function)
assert tool.tool_type == TOOL_TYPE_FUNCTION
```

### Type Checking

```python
from dapr_agents.types import ToolType, TOOL_TYPE_FUNCTION

def process_tool(tool_type: ToolType) -> None:
    """Process a tool based on its type."""
    if tool_type == TOOL_TYPE_FUNCTION:
        print("Processing function tool")
    elif tool_type == TOOL_TYPE_MCP:
        print("Processing MCP tool")
```

## Benefits

1. **Type Safety**: IDE autocomplete and type checkers can validate tool types
2. **No Magic Strings**: Constants prevent typos like `"fucntion"` or `"Function"`
3. **Centralized Definition**: Change tool type values in one place
4. **Better Refactoring**: Find all usages easily with "Find References"
5. **Self-Documenting**: Constants are more readable than string literals

## Location

All constants are defined in:
- **Module**: `dapr_agents/registry/metadata.py`
- **Package**: `dapr_agents/registry/__init__.py`

## Framework Independence

These constants are framework-agnostic and can be used with any agent framework that adopts the metadata types.



