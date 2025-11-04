# Agent Registry Module

The `dapr_agents.registry` module provides framework-agnostic types for agent metadata storage and registry management with Dapr state stores.

## Purpose

This module houses metadata types and registry management that are independent of any specific agent framework implementation. This design allows:

- **Framework Independence**: Types can be used with any agent framework
- **Reusability**: Easy to adopt in other projects
- **Consistency**: Maintains the same metadata structure across different implementations
- **Separation of Concerns**: Registry types are isolated from agent implementation details
- **Uniqueness Enforcement**: Ensures agent names are unique within and across processes

## Two Registry Systems

The Dapr Agents framework uses two complementary registries:

### 1. **Agent Registry** (Metadata Discovery)
- **Purpose**: Store comprehensive agent metadata for discovery and introspection
- **Key**: `"agents-registry"`
- **Managed by**: `Registry` class with `DaprClient`
- **Content**: Full `AgentMetadata` with tools, components, capabilities
- **Use case**: Agent discovery, capability queries, debugging, monitoring
- **State Store Prefix**: Should use `name` or `namespace` to enable cross-application sharing

### 2. **Team Registry** (Runtime Addressing)
- **Purpose**: Enable pub/sub messaging between agents in multi-agent systems
- **Key**: `"agents:{team_name}"` (e.g., `"agents:default"`)
- **Managed by**: `StateStoreService` via `AgentComponents`
- **Content**: Minimal metadata for pub/sub addressing (name, topic, pubsub_name)
- **Use case**: Orchestrators discovering and messaging team agents at runtime
- **State Store Prefix**: Can use `appid` (default) for single-app teams, or `name`/`namespace` for cross-app teams

## Module Contents

### Metadata Types (`metadata.py`)

#### Component Models
- **`ComponentBase`**: Base class for Dapr component references
- **`StateStoreComponent`**: State store component reference (name, usage, parameters)
- **`PubSubComponent`**: Pub/Sub component reference with topic name
- **`BindingComponent`**: Input/Output binding component reference
- **`SecretStoreComponent`**: Secret store component reference
- **`ConfigurationStoreComponent`**: Configuration store component reference
- **`ConversationComponent`**: Conversation component reference

#### Main Models
- **`ToolDefinition`**: Tool metadata for agent registration
- **`ComponentMappings`**: Typed references to Dapr components grouped by category
- **`AgentMetadata`**: Complete agent registration metadata structure

### Agent Category Constants

- **`AgentCategory`**: Type alias for agent categories
  - `"agent"`: Standard conversational/task agents (e.g., `Agent`)
  - `"durable-agent"`: Workflow-based agents with durable execution (e.g., `DurableAgent`)
  - `"orchestrator"`: Multi-agent coordination agents (e.g., `LLMOrchestrator`, `RandomOrchestrator`)

### Tool Type Constants

- **`TOOL_TYPE_FUNCTION`**: Python function tools
- **`TOOL_TYPE_MCP`**: Model Context Protocol tools
- **`TOOL_TYPE_AGENT`**: Agent-as-tool
- **`TOOL_TYPE_UNKNOWN`**: Unspecified tool type
- **`ToolType`**: Type alias for all valid tool types

### Registry Management (`registry.py`)

The `Registry` class provides agent metadata management:
- **Persistent storage** of agent metadata in Dapr state stores
- **Uniqueness enforcement** for agent names (local cache + remote store)
- **Optimistic concurrency** with retry logic using etags
- **Idempotent registration** - same agent metadata can be re-registered without error
- **Conflict detection** - different agent metadata with same name raises error
- **Auto-detection** of Dapr app ID from metadata endpoint

### Registry Mixin (`registry_mixin.py`)

The `RegistryMixin` provides unified registration interface:
- **Abstract method** `_build_agent_metadata()` for subclasses to implement
- **Automatic registration** during agent initialization
- **DaprClient management** for registry operations
- **Error handling** with graceful fallback

## State Store Configuration

### Recommended Component Definitions

The two registries should use separate state store component definitions with different `keyPrefix` strategies:

#### Agent Registry State Store

Use `name` or `namespace` prefix to allow agents from different applications to register in the same registry:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: agent-registry-store
  namespace: default
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: keyPrefix
    value: name  # Enables cross-application sharing
```

With this configuration:
- Key `"agents-registry"` is stored as `"agent-registry-store||agents-registry"`
- All applications using this component can access the same agent registry
- Agents from different app-ids can discover each other

#### Team Registry State Store

Use `appid` (default) for single-application teams, or `name`/`namespace` for multi-application teams:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: team-registry-store
  namespace: default
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: keyPrefix
    value: name  # For cross-application teams
    # Or use: appid (default, single-app teams)
    # Or use: namespace (namespace-scoped teams)
```

For more information on state sharing strategies, see the [Dapr documentation on sharing state between applications](https://docs.dapr.io/developing-applications/building-blocks/state-management/howto-share-state/).

### Key Prefix Strategy Comparison

| Strategy | Agent Registry | Team Registry | Use Case |
|----------|---------------|---------------|----------|
| `name` | ✅ Recommended | ✅ Multi-app teams | All apps sharing component name can access |
| `namespace` | ✅ Recommended | ✅ Namespace-scoped teams | Apps in same namespace can share |
| `appid` | ❌ Too restrictive | ✅ Single-app teams | Only one app-id can access (default) |
| `none` | ⚠️ Too permissive | ⚠️ Rarely needed | All apps share (no scoping) |

## Usage

### Creating Metadata

```python
from dapr_agents.registry.metadata import (
    AgentMetadata,
    ToolDefinition,
    ComponentMappings,
    StateStoreComponent,
    PubSubComponent,
    TOOL_TYPE_FUNCTION,
)

# Create tool definition
tool = ToolDefinition(
    name="WeatherTool",
    description="Get weather information",
    tool_type=TOOL_TYPE_FUNCTION
)

# Create component mappings with typed references
components = ComponentMappings(
    state_stores={
        "memory": StateStoreComponent(
            name="statestore",
            usage="Conversation and long-term memory store"
        ),
        "agent_registry": StateStoreComponent(
            name="agent-registry-store",
            usage="Agent metadata discovery registry"
        ),
        "team_registry": StateStoreComponent(
            name="team-registry-store",
            usage="Team pub/sub addressing registry"
        ),
    },
    pubsub_components={
        "message_bus": PubSubComponent(
            name="pubsub",
            usage="Primary pub/sub component for agent messaging",
            topic_name="WeatherAgent"
        )
    },
)

# Create agent metadata
metadata = AgentMetadata(
    name="WeatherAgent",
    role="Weather Assistant",
    goal="Provide weather information",
    tools=[tool],
    components=components,
    system_prompt="You are a weather assistant",
    agent_id="weather-agent-instance-123",
    agent_class="Agent",  # Technical implementation class
    agent_category="agent",  # Functional category
    agent_framework="dapr-agents",
    dapr_app_id="weather-app",  # Auto-detected if not provided
    namespace="default",  # Optional
    sub_agents=[],  # For orchestrators: list of managed agent names
)

# Serialize for registry storage
registry_data = metadata.model_dump_for_registry()
```

### Using the Agent Registry (Manual)

```python
from dapr.clients import DaprClient
from dapr_agents.registry.registry import Registry

# Initialize registry (for agent metadata discovery)
with DaprClient() as client:
    registry = Registry(
        client=client,
        store_name="statestore",
        store_key="agents-registry"  # Fixed key for agent metadata
    )
    
    # Register an agent (with uniqueness enforcement)
    registry.register_agent(
        agent_name="WeatherAgent",
        agent_metadata=metadata.model_dump_for_registry(),
        agent_identity="unique-instance-id"
    )
```

### Automatic Registration (Recommended)

Agents automatically register themselves when using `RegistryMixin`:

```python
from dapr_agents.agents.standalone import Agent
from dapr_agents.agents.configs import AgentRegistryConfig, AgentStateConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreService

# IMPORTANT: Use separate state store components for each registry
# with appropriate keyPrefix settings (see State Store Configuration above)

# Agent registry store (with keyPrefix: name or namespace)
agent_registry_store = StateStoreService(store_name="agent-registry-store")
agent_registry_config = AgentRegistryConfig(store=agent_registry_store)

# Team registry store (can use keyPrefix: appid or name)
team_registry_store = StateStoreService(store_name="team-registry-store")
team_registry_config = AgentRegistryConfig(store=team_registry_store)

# Example 1: Agent registry only (metadata discovery)
agent = Agent(
    name="WeatherAgent",
    role="Weather Assistant",
    goal="Provide weather information",
    agent_registry_config=agent_registry_config,  # For agent metadata discovery
)

# Example 2: Team registry only (pub/sub addressing for orchestrators)
agent_team_only = Agent(
    name="TeamAgent",
    role="Team Member",
    registry_config=team_registry_config,  # For team pub/sub addressing
)

# Example 3: Both registries (recommended for production)
agent_full = Agent(
    name="FullAgent",
    role="Full Featured Agent",
    agent_registry_config=agent_registry_config,  # Agent metadata discovery
    registry_config=team_registry_config,  # Team pub/sub addressing
)
```

### Team Registry for Orchestrators

Orchestrators use the team registry to discover and message agents:

```python
from dapr_agents.agents.orchestrators.llm.orchestrator import LLMOrchestrator

# Orchestrator can query team agents via team registry
orchestrator = LLMOrchestrator(
    name="TaskOrchestrator",
    registry_config=team_registry_config,  # Team registry for pub/sub addressing
    agent_registry_config=agent_registry_config,  # Optional: agent metadata discovery
    # ... other config
)

# Inside orchestrator workflows/activities:
agents = orchestrator.list_team_agents(team="default", include_self=False)
# Returns: {"WeatherAgent": {"name": "WeatherAgent", "topic_name": "...", ...}}

# Orchestrator can then send messages to discovered agents via pub/sub
```

## Key Features

### Agent Categorization

Agents are categorized by functionality using the `_agent_category_override` class attribute:

```python
# AgentBase - standard conversational agents
class AgentBase:
    _agent_category_override = None  # Defaults to "agent"

# DurableAgent - workflow-based agents
class DurableAgent(AgentBase):
    _agent_category_override = "durable-agent"

# OrchestratorBase - multi-agent coordinators
class OrchestratorBase:
    _agent_category_override = "orchestrator"
```

Categories help with:
- **Discovery**: Query agents by category (e.g., find all orchestrators)
- **Monitoring**: Track different agent types separately
- **Routing**: Different handling based on agent capabilities

### Uniqueness Enforcement

The agent registry enforces name uniqueness at two levels:

1. **Local Process Cache**: Fast check using in-memory registry (`Registry._registered_agent_names`)
2. **Remote State Store**: Cross-process check using Dapr state store with optimistic concurrency

### Idempotent Registration

Re-registering an agent with identical metadata succeeds without error. This allows:
- Agent restarts without manual cleanup
- Multiple registration calls with same configuration
- Simplified deployment scenarios

### Conflict Detection

Attempting to register a different agent with an existing name raises a `ValueError`:
```python
# First registration succeeds
register_agent(name="MyAgent", metadata=metadata_v1)

# Same metadata - succeeds (idempotent)
register_agent(name="MyAgent", metadata=metadata_v1)

# Different metadata - raises ValueError
register_agent(name="MyAgent", metadata=metadata_v2)  # ❌ Conflict!
```

### Dual Registry Architecture

The two registries serve different purposes and are accessed differently:

| Feature | Agent Registry | Team Registry |
|---------|---------------|---------------|
| **Key** | `"agents-registry"` | `"agents:{team}"` |
| **Purpose** | Metadata discovery | Runtime pub/sub addressing |
| **Managed By** | `Registry` + `DaprClient` | `StateStoreService` |
| **Content** | Full `AgentMetadata` | Minimal pub/sub info |
| **Used By** | Monitoring, debugging | Orchestrators messaging agents |
| **When** | Agent initialization | Workflow/activity runtime |
| **State Store Component** | `agent-registry-store` | `team-registry-store` |
| **Recommended keyPrefix** | `name` or `namespace` | `appid` or `name` |
| **Sharing Scope** | Cross-application | Single or multi-app |

### Why Separate State Store Components?

Using separate Dapr state store components for each registry allows:

1. **Different keyPrefix strategies**: Agent registry needs cross-app sharing (`name`), while team registry might be app-scoped (`appid`)
2. **Independent scaling**: Different performance/consistency requirements
3. **Separate backends**: Agent registry could use a different storage technology than team registry
4. **Access control**: Different security policies per registry type
5. **Clear separation of concerns**: Metadata discovery vs runtime addressing

Without separate components, you cannot configure different `keyPrefix` values for the same physical state store.

## Implementation Details

### Optimistic Concurrency

The registry uses etags for optimistic concurrency control:
- Reads current state with etag
- Validates uniqueness constraints
- Writes with etag transaction (fails if state changed)
- Retries on conflict (up to 20 attempts with exponential backoff)

### Component Categories

Components are organized by Dapr component type:
- `state_stores`: State store components (memory, registry, workflow)
- `pubsub_components`: Pub/Sub components (message bus, notifications)
- `binding_components`: Input/Output bindings
- `secret_stores`: Secret store components
- `configuration_stores`: Configuration store components

Each component reference includes:
- `name`: Dapr component name
- `usage`: Description of how the agent uses it
- `parameters`: Optional runtime parameters (dict)
- Type-specific fields (e.g., `topic_name` for PubSubComponent)

## Testing

The registry module has comprehensive test coverage:
- **`metadata.py`**: 100% coverage ✅
- **`registry.py`**: 84% coverage ✅

Integration tests with real Dapr runtime validate:
- Agent metadata persistence
- Tool definitions storage
- Component mappings structure
- Name uniqueness enforcement
- Idempotent re-registration
- DurableAgent-specific fields
- Error handling

Run tests with:
```bash
DAPR_INTEGRATION=1 pytest tests/agents/test_agent_registration_integration.py -v
```

## Documentation

This README serves as the complete documentation for the agent registry module, including:
- Metadata structure and types
- Tool type constants
- Dual registry architecture
- State store configuration
- Usage examples and best practices

## Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Agent/Orchestrator Classes                              │
│ (AgentBase, DurableAgent, OrchestratorBase)             │
└────────────────┬────────────────────────────────────────┘
                 │ inherits
                 ▼
┌─────────────────────────────────────────────────────────┐
│ RegistryMixin                                           │
│ • _build_agent_metadata() [abstract]                    │
│ • _register_agent_metadata()                            │
└────────────────┬────────────────────────────────────────┘
                 │ uses
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Registry                                                │
│ • register_agent()                                      │
│ • Uniqueness enforcement                                │
│ • Optimistic concurrency                                │
└────────────────┬────────────────────────────────────────┘
                 │ uses
                 ▼
┌─────────────────────────────────────────────────────────┐
│ DaprClient                                              │
│ • State store operations                                │
│ • Metadata endpoint access                              │
└─────────────────────────────────────────────────────────┘
```

### Registration Flow

1. **Agent Initialization**: Agent/orchestrator class is instantiated
2. **Metadata Building**: `_build_agent_metadata()` extracts config into `AgentMetadata`
3. **Registration Call**: `_register_agent_metadata()` creates `DaprClient` and `Registry`
4. **Persistence**: `Registry.register_agent()` stores metadata with uniqueness checks
5. **Team Registration**: Separately, `register_agentic_system()` adds to team registry

### Framework Agnosticism

While these types are framework-agnostic, the `Registry` class performs the actual interaction with the Dapr state store. Agent implementations delegate to this class to:
- Build metadata from agent configuration
- Persist metadata to state store
- Enforce name uniqueness
- Handle registration errors

The separation between metadata types and registry logic allows easy adoption in other frameworks while maintaining consistent storage format.

### When to Use Which Registry

**Use Agent Registry (`agents-registry`) when:**
- Implementing monitoring/observability dashboards
- Building agent discovery tools
- Debugging agent configurations
- Querying agent capabilities and tools
- Tracking agent metadata changes
- Discovering agents across multiple applications/app-ids

**Use Team Registry (`agents:{team}`) when:**
- Orchestrators need to discover available agents
- Sending pub/sub messages between agents
- Dynamic team composition at runtime
- Broadcast messages to team members
- Agent-to-agent communication
- Agents run in the same application/namespace

## Component Configuration Examples

```yaml

# components/agent-registry-store.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: agent-registry-store
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis:6379
  - name: keyPrefix
    value: name

# components/team-registry-store.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: team-registry-store
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis:6379
  - name: keyPrefix
    value: name  # or appid for single-app teams
```
