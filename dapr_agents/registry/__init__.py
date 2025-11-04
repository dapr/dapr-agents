"""
Agent registry module.

This module provides framework-agnostic types for agent metadata storage
and registry management.
"""

from dapr_agents.registry.metadata import (
    ToolDefinition,
    ComponentMappings,
    AgentMetadata,
    ComponentBase,
    StateStoreComponent,
    PubSubComponent,
    BindingComponent,
    SecretStoreComponent,
    ConfigurationStoreComponent,
    ToolType,
    AgentCategory,
    TOOL_TYPE_FUNCTION,
    TOOL_TYPE_MCP,
    TOOL_TYPE_AGENT,
    TOOL_TYPE_UNKNOWN,
)
from dapr_agents.registry.registry import Registry
from dapr_agents.registry.registry_mixin import RegistryMixin

__all__ = [
    "ToolDefinition",
    "ComponentMappings",
    "AgentMetadata",
    "ComponentBase",
    "StateStoreComponent",
    "PubSubComponent",
    "BindingComponent",
    "SecretStoreComponent",
    "ConfigurationStoreComponent",
    "ToolType",
    "AgentCategory",
    "TOOL_TYPE_FUNCTION",
    "TOOL_TYPE_MCP",
    "TOOL_TYPE_AGENT",
    "TOOL_TYPE_UNKNOWN",
    "Registry",
    "RegistryMixin",
]
