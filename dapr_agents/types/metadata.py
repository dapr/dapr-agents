"""
Agent metadata types for registry storage.

These types are framework-agnostic and can be used with any agent framework
to maintain consistent metadata structure in agent registries.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

# Tool type constants
TOOL_TYPE_FUNCTION = "function"
TOOL_TYPE_MCP = "mcp"
TOOL_TYPE_AGENT = "agent"
TOOL_TYPE_UNKNOWN = "unknown"

# Type alias for tool types
ToolType = Literal[
    TOOL_TYPE_FUNCTION, TOOL_TYPE_MCP, TOOL_TYPE_AGENT, TOOL_TYPE_UNKNOWN
]


class ToolDefinition(BaseModel):
    """
    Tool metadata for agent registration.

    Represents a tool available to an agent, including its type and description.
    Used for storing tool information in the agent registry.

    Attributes:
        name: The name of the tool
        description: A brief description of the tool's functionality
        tool_type: The type of tool (function, mcp, agent, or unknown)
    """

    name: str = Field(..., description="The name of the tool")
    description: str = Field(
        ..., description="A brief description of the tool's functionality"
    )
    tool_type: ToolType = Field(
        ...,
        description=f"The type of tool: '{TOOL_TYPE_FUNCTION}' for Python functions, '{TOOL_TYPE_MCP}' for MCP tools, '{TOOL_TYPE_AGENT}' for agent tools, or '{TOOL_TYPE_UNKNOWN}'",
    )


class ComponentMappings(BaseModel):
    """
    Component mappings for agent registration.

    Tracks the component names used by an agent for various storage or messaging needs.
    This is framework-specific but the structure is generic enough for reuse.

    Attributes:
        memory_component: Component name for agent memory/conversation history
        registry_component: Component name for agent registry
        workflow_component: Component name for workflow state
        pubsub_component: Optional component name for pubsub (used by agents that support messaging)
    """

    memory_component: str = Field(
        ..., description="Component name for agent memory/conversation history"
    )
    registry_component: str = Field(
        ..., description="Component name for agent registry"
    )
    workflow_component: str = Field(
        ..., description="Component name for workflow state"
    )
    pubsub_component: str | None = Field(None, description="Component name for pubsub")


class AgentMetadata(BaseModel):
    """
    Agent registration metadata.

    This model defines the complete metadata structure stored in the agent registry.
    It includes core agent properties, tool definitions, and component mappings.

    This structure is designed to be framework-agnostic, allowing it to be used
    with different agent frameworks while maintaining consistency.

    Attributes:
        name: Agent's unique name
        role: Agent's role description
        goal: Agent's main objective
        tool_choice: Strategy for tool selection ('auto', 'required', 'none')
        instructions: List of instructions guiding the agent
        orchestrator: Whether this agent is an orchestrator
        tools: List of tool definitions available to the agent
        components: Component mappings used by the agent
    """

    name: str = Field(..., description="The agent's unique name")
    role: str | None = Field(None, description="The agent's role in the system")
    goal: str | None = Field(None, description="The agent's main objective")
    tool_choice: str | None = Field(None, description="Strategy for tool selection")
    instructions: List[str] | None = Field(
        None, description="Instructions guiding the agent's tasks"
    )
    tools: List[ToolDefinition] = Field(
        default_factory=list, description="Tools available to the agent"
    )
    components: ComponentMappings = Field(..., description="Component mappings")
    system_prompt: str = Field(..., description="The agent's system prompt")

    agent_id: str = Field(..., description="The agent's unique identifier")
    agent_framework: str = Field(
        default="dapr-agents", description="The agent framework name"
    )
    agent_type: str = Field(..., description="The agent type within the framework")

    def model_dump_for_registry(self) -> Dict[str, Any]:
        """
        Serialize metadata for registry storage.

        Excludes None values and uses JSON-compatible serialization mode.

        Returns:
            Dict suitable for storage in the agent registry
        """
        return self.model_dump(exclude_none=True, mode="json")
