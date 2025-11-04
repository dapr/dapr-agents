"""
Agent metadata types for registry storage.

These types are framework-agnostic and can be used with any agent framework
to maintain consistent metadata structure in agent registries.
"""

from pydantic import BaseModel, Field, ConfigDict
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

# Agent category type
AgentCategory = Literal["agent", "durable-agent", "orchestrator"]


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


class ComponentBase(BaseModel):
    """Base definition for a Dapr component reference."""

    name: str = Field(..., description="Dapr component name as declared in the runtime")
    usage: str = Field(
        ..., description="Brief description of how the agent uses this component"
    )
    parameters: Dict[str, Any] | None = Field(
        default=None,
        description="Optional runtime parameters supplied when invoking or instantiating the component",
    )

    model_config = ConfigDict(extra="ignore")


class StateStoreComponent(ComponentBase):
    """State store component reference."""


class PubSubComponent(ComponentBase):
    """Pub/Sub component reference."""

    topic_name: str = Field(..., description="Topic name for the pub/sub component")


class BindingComponent(ComponentBase):
    """Binding component reference."""


class SecretStoreComponent(ComponentBase):
    """Secret store component reference."""


class ConfigurationStoreComponent(ComponentBase):
    """Configuration store component reference."""


class ConversationComponent(ComponentBase):
    """Conversation component reference."""


class ComponentMappings(BaseModel):
    """
    Typed references to the Dapr components an agent relies on.

    Components are grouped by Dapr category and keyed by a logical usage label
    (for example: "memory", "registry", "workflow", "default", "notifications").
    """

    state_stores: Dict[str, StateStoreComponent] = Field(
        default_factory=dict,
        description="State store components keyed by usage identifier",
    )
    pubsub_components: Dict[str, PubSubComponent] = Field(
        default_factory=dict,
        description="Pub/Sub components keyed by usage identifier",
    )
    binding_components: Dict[str, BindingComponent] = Field(
        default_factory=dict,
        description="Input/Output binding components keyed by usage identifier",
    )
    secret_stores: Dict[str, SecretStoreComponent] = Field(
        default_factory=dict,
        description="Secret store components keyed by usage identifier",
    )
    configuration_stores: Dict[str, ConfigurationStoreComponent] = Field(
        default_factory=dict,
        description="Configuration store components keyed by usage identifier",
    )


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
        tools: List of tool definitions available to the agent
        components: Component mappings used by the agent
        agent_class: Agent implementation class name
        agent_category: Functional category (agent, durable-agent, or orchestrator)
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

    system_prompt: str = Field(default="", description="The agent's system prompt")

    agent_id: str = Field(..., description="The agent's unique identifier")
    agent_framework: str = Field(
        default="dapr-agents", description="The agent framework name"
    )
    agent_class: str = Field(
        ...,
        description="Agent implementation class (e.g., 'Agent', 'DurableAgent', 'LLMOrchestrator')",
    )
    agent_category: AgentCategory = Field(
        ...,
        description="Functional category: 'agent' for standard agents, 'durable-agent' for workflow-based agents, 'orchestrator' for multi-agent coordination",
    )

    dapr_app_id: str | None = Field(
        None,
        description="The Dapr app ID (can be retrieved from Dapr metadata endpoint)",
    )
    namespace: str | None = Field(
        None, description="The app namespace/project ID (optional)"
    )
    sub_agents: List[str] = Field(
        default_factory=list,
        description="Child agents managed by this agent/orchestrator",
    )

    model_config = ConfigDict(extra="allow")

    def model_dump_for_registry(self) -> Dict[str, Any]:
        """
        Serialize metadata for registry storage.

        Excludes None values and uses JSON-compatible serialization mode.

        Returns:
            Dict suitable for storage in the agent registry
        """
        return self.model_dump(exclude_none=True, mode="json")
