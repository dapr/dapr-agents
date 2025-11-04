# Agent Registry Metadata

## Overview

The agent registry stores metadata about registered agents, including their tools, component mappings, and configuration. This metadata enables agent discovery, introspection, and coordination in multi-agent systems.

The metadata types are defined in `dapr_agents/registry/metadata.py` and are framework-agnostic for reusability across different agent implementations.

## Metadata Structure

### Tool Definitions

Each agent's registered tools include:

- **name**: The tool's identifier
- **description**: Brief description of the tool's functionality  
- **tool_type**: The type of tool, one of:
  - `"function"` - Python function tools
  - `"mcp"` - Model Context Protocol tools
  - `"agent"` - Agent-as-tool (placeholder for future use)
  - `"unknown"` - Unspecified tool type

**Example:**
```python
{
    "name": "WeatherTool",
    "description": "Get current weather information",
    "tool_type": "function"
}
```

### Component Mappings

Component mappings reference Dapr components by category, keyed by a logical usage label:

- **state_stores**: Map of logical usage keys (e.g., `"memory"`, `"registry"`, `"workflow"`) to `StateStoreComponent` definitions
- **pubsub_components**: Map of usage keys to `PubSubComponent` definitions (e.g., `"message_bus"`)
- **binding_components**: Map of usage keys to `BindingComponent` definitions
- **secret_stores**: Map of usage keys to `SecretStoreComponent` definitions
- **configuration_stores**: Map of usage keys to `ConfigurationStoreComponent` definitions

Each component definition captures the runtime component name, a brief usage description, and optional component-specific metadata (such as provider or driver type).

**Example:**
```json
{
    "state_stores": {
        "memory": {
            "name": "statestore",
            "usage": "Conversation memory store"
        },
        "registry": {
            "name": "registry-store",
            "usage": "Agent registry store"
        },
        "workflow": {
            "name": "statestore",
            "usage": "Workflow state store"
        }
    },
    "pubsub_components": {
        "message_bus": {
            "name": "pubsub",
            "usage": "Primary message bus"
        }
    },
    "binding_components": {},
    "secret_stores": {},
    "configuration_stores": {}
}
```

### Complete Agent Metadata

The full metadata structure stored in the registry for each agent includes:

**Core Properties:**
- **name**: Agent's unique identifier (required)
- **role**: Agent's role description
- **goal**: Agent's main objective
- **tool_choice**: Strategy for tool selection (e.g., "auto", "required", "none")
- **instructions**: List of instructions guiding the agent
- **agent_id**: Unique identifier for the agent (default is the agent's name, but it can be overridden)
- **agent_framework**: Framework name (e.g., "dapr-agents")
- **agent_type**: Agent class type (e.g., "Agent", "DurableAgent")

**Collections:**
- **tools**: Array of tool definitions (see above)
- **components**: Component mappings (see above)

**Additional Properties:**
Specific agent types may include additional metadata fields:
- **agent_topic_name**: Topic name for durable agents
- **message_bus_name**: Pub/sub component name
- **broadcast_topic_name**: Topic for broadcasting in workflows
- **orchestrator_topic_name**: Topic for orchestrator communication

**Example:**
```json
{
    "name": "WeatherAgent",
    "role": "Weather Assistant",
    "goal": "Provide accurate weather information",
    "tool_choice": "auto",
    "instructions": ["Always check latest data"],
    "agent_id": "WeatherAgent",
    "agent_framework": "dapr-agents",
    "agent_type": "Agent",
    "tools": [
        {
            "name": "GetWeather",
            "description": "Fetch weather data",
            "tool_type": "function"
        }
    ],
    "components": {
        "state_stores": {
            "memory": {
                "name": "statestore",
                "usage": "Conversation memory store"
            },
            "registry": {
                "name": "registry-store",
                "usage": "Agent registry store"
            },
            "workflow": {
                "name": "statestore",
                "usage": "Workflow state store"
            }
        },
        "pubsub_components": {
            "message_bus": {
                "name": "pubsub",
                "usage": "Primary message bus"
            }
        },
        "binding_components": {},
        "secret_stores": {},
        "configuration_stores": {}
    },
    "agent_topic_name": "weather-topic",
    "message_bus_name": "pubsub"
}
```

## Agent Registration

When an agent is created, its metadata is automatically collected and stored in the registry:

```python
from dapr_agents.agents import Agent
from dapr_agents.tool.base import AgentTool

# Create tools
def my_tool(input: str) -> str:
    """My custom tool."""
    return f"Processed: {input}"

tool = AgentTool.from_func(my_tool)

# Create agent - metadata is automatically registered
agent = Agent(
    name="MyAgent",
    role="Assistant",
    goal="Help users",
    tools=[tool],
    memory_store=MemoryStore(name="statestore")
)

# Metadata is automatically captured:
# - Tool definitions (name, description, type)
# - Component mappings (state stores, pubsub)
# - Agent properties (name, role, goal, etc.)
# - Serialized and stored in registry
```

### Registry Helper

Agent registration is handled by the `Registry` helper in `dapr_agents/registry/registry.py`.

- Tracks in-process agent names to avoid duplicates
- Writes metadata to the configured Dapr state store
- Supports idempotent re-registration and metadata updates for the same agent identity

`AgentBase` and `AgenticWorkflow` delegate their registration to this helper so the registry logic stays reusable and framework-agnostic.

## Name Uniqueness

Agent names must be unique:
- **Within a process**: No two agents with the same name in the same application
- **Across processes**: No two agents with the same name for the same `app_id` (even across different sidecars)
- **Idempotent re-registration**: The same agent instance can re-register without error

## Retrieving Metadata from Registry

```python
# Get all agents from registry
agents_metadata = agent.get_agents_metadata()

for agent_name, metadata in agents_metadata.items():
    print(f"Agent: {metadata['name']}")
    print(f"Role: {metadata['role']}")
    print(f"Tools: {len(metadata['tools'])}")
    memory_store = metadata['components']['state_stores'].get('memory')
    if memory_store:
        print(f"Memory component: {memory_store['name']}")
    
    # Access tool information
    for tool in metadata['tools']:
        print(f"  - {tool['name']}: {tool['tool_type']}")
```

## Use Cases

Agent registry metadata enables:

1. **Agent Discovery**: Find available agents and their capabilities
2. **Tool Introspection**: Understand what tools each agent has access to
3. **Component Tracking**: Know which Dapr components are being used
4. **Multi-Agent Coordination**: Orchestrators can discover and communicate with other agents
5. **Monitoring & Debugging**: Inspect agent configuration and dependencies

## Examples

See `/examples/strongly_typed_metadata_example.py` for a complete working example of the metadata types and usage.
