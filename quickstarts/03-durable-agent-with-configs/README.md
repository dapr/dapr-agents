# Durable Agent with Advanced Configurations

This quickstart demonstrates how to create a **Durable Agent** using **Agent Configuration Classes** with Dapr Agents. You'll learn how to leverage configuration objects for fine-grained control over pub/sub messaging, state management, memory, registry, and agent profiles—enabling production-ready, stateful workflows with proper separation of concerns.

## What You'll Learn

- How to use **Agent Configuration Classes** for modular setup
- Configure **Pub/Sub messaging** for event-driven agent communication
- Set up **State persistence** with custom storage backends
- Implement **Agent Registry** for multi-agent coordination
- Configure **Conversation Memory** with Dapr state stores
- Define **Agent Profiles** for consistent prompting and behavior
- Deploy agents that respond to pub/sub events using **workflow-backed handlers**

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Dapr CLI and Docker
- OpenAI API key (or compatible LLM provider)

## Key Differences: Simple vs. Configured Agents

### Simple Agent (03-durable-agent-tool-call)
```python
# Minimal configuration - uses defaults
agent = DurableAgent(
    role="Weather Assistant",
    name="Stevie",
    goal="Help humans get weather info.",
    instructions=["Be helpful"],
    tools=tools,
)
agent.start()

# Direct execution via AgentRunner
runner = AgentRunner()
result = await runner.run(agent, payload={"task": prompt})
```

### Configured Agent (This Quickstart)
```python
# Explicit configuration with full control
agent = DurableAgent(
    profile=AgentProfileConfig(...),      # Structured prompting
    pubsub=AgentPubSubConfig(...),        # Pub/sub messaging
    state=AgentStateConfig(...),          # State persistence
    registry=AgentRegistryConfig(...),    # Agent registry
    memory=AgentMemoryConfig(...),        # Conversation memory
    llm=llm,
    tools=tools,
)
agent.start()

# Event-driven execution via pub/sub
runner = AgentRunner()
runner.register_routes(agent)  # Registers pub/sub handlers
await wait_for_shutdown()      # Waits for events
```

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The quickstart uses multiple Dapr components for state, memory, registry, and messaging:

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

2. When running the examples with Dapr, use the helper script to resolve environment variables:
```bash
# Get the environment variables from the .env file:
export $(grep -v '^#' ../../.env | xargs)

# Create a temporary resources folder with resolved environment variables
temp_resources_folder=$(../resolve_env_templates.py ./components)

# Run your dapr command with the temporary resources
dapr run \
  --app-id weather-agent \
  --resources-path $temp_resources_folder \
  -- python app.py

# Clean up when done
rm -rf $temp_resources_folder
```

### Option 2: Direct Component Configuration

Update the `key` in [components/openai.yaml](components/openai.yaml):
```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: openai
spec:
  type: conversation.openai
  metadata:
    - name: key
      value: "YOUR_OPENAI_API_KEY"
```

### Required Components

This quickstart requires these Dapr components (included in `components/` directory):

- `openai.yaml`: LLM conversation component
- `workflowstate.yaml`: Workflow state storage
- `memorystore.yaml`: Conversation memory storage
- `agentregistrystore.yaml`: Agent registry storage
- `messagepubsub.yaml`: Pub/sub messaging for agent communication

Make sure Dapr is initialized on your system:

```bash
dapr init
```

## Understanding Agent Configuration Classes

Agent Configuration Classes provide a structured, type-safe way to configure durable agents. They are defined in [`dapr_agents/agents/configs.py`](../../dapr_agents/agents/configs.py) and offer the following benefits:

### 1. **AgentProfileConfig** - Agent Identity & Prompting

Defines the agent's persona, role, and behavior:

```python
profile = AgentProfileConfig(
    name="Weather Agent",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Always answer weather questions directly.",
        "Provide concise summaries.",
    ],
    style_guidelines=["Be professional", "Be concise"],
    template_format="jinja2",  # Or "simple"
)
```

**Benefits:**
- Consistent prompting across agent instances
- Modular instruction management
- Support for Jinja2 templates
- Clear separation of concerns

### 2. **AgentPubSubConfig** - Event-Driven Messaging

Configures pub/sub topics for agent communication:

```python
pubsub = AgentPubSubConfig(
    pubsub_name="messagepubsub",
    agent_topic="weather.requests",      # Direct messages to this agent
    broadcast_topic="agents.broadcast",  # Team-wide broadcasts
)
```

**Benefits:**
- Event-driven architecture
- Async, non-blocking communication
- Multi-agent coordination via broadcast
- Decoupled from HTTP endpoints

### 3. **AgentStateConfig** - State Persistence

Manages workflow state storage with automatic schema handling:

```python
state = AgentStateConfig(
    store=StateStoreService(
        store_name="workflowstatestore",
        key_prefix="weather:"
    ),
    default_state=None,  # Optional default state
)
```

**Benefits:**
- Automatic state schema selection (per agent type)
- Durable workflow execution
- State recovery after failures
- Custom state models supported via hooks

**Advanced:** Supports custom state models with hooks:
- `entry_factory`: Create custom workflow entries
- `message_coercer`: Transform message dictionaries
- `entry_container_getter`: Locate instance containers

### 4. **AgentRegistryConfig** - Multi-Agent Coordination

Enables agent discovery and metadata sharing:

```python
registry = AgentRegistryConfig(
    store=StateStoreService(store_name="agentregistrystore"),
    team_name="weather",  # Group agents into teams
)
```

**Benefits:**
- Dynamic agent discovery
- Metadata sharing between agents
- Team-based organization
- Handoff coordination support

### 5. **AgentMemoryConfig** - Conversation History

Configures conversation memory storage:

```python
memory = AgentMemoryConfig(
    store=ConversationDaprStateMemory(
        store_name="memorystore",
        session_id="weather-agent-session",
    )
)
```

**Benefits:**
- Persistent conversation history
- Multi-turn dialogue support
- Session management
- Pluggable memory backends

### 6. **AgentExecutionConfig** - Runtime Control

Fine-tune agent execution behavior:

```python
execution = AgentExecutionConfig(
    max_iterations=10,      # Maximum LLM call iterations
    tool_choice="auto",     # "auto", "required", or specific tool
)
```

**Benefits:**
- Prevent infinite loops
- Control tool selection strategy
- Runtime behavior customization

### 7. **WorkflowGrpcOptions** - Large Payload Support

Configure gRPC limits for large data transfers:

```python
workflow_grpc = WorkflowGrpcOptions(
    max_send_message_length=16 * 1024 * 1024,     # 16 MB
    max_receive_message_length=16 * 1024 * 1024,  # 16 MB
)
```

**Benefits:**
- Handle large tool outputs
- Process documents/images
- Avoid message size errors

## Running the Example

### Terminal 1: Start the Agent Service

The agent listens for pub/sub messages:

```bash
dapr run \
  --app-id weather-agent \
  --resources-path ./components \
  -- python app.py
```

The agent will:
1. Register workflows with the Dapr workflow runtime
2. Subscribe to `weather.requests` topic
3. Listen for incoming messages
4. Execute workflows in response to events

### Terminal 2: Send a Message to Trigger the Agent

Use the client to publish a message:

```bash
dapr run \
  --app-id weather-client \
  --resources-path ./components \
  -- python message_client.py
```

This publishes a message to the `weather.requests` topic, triggering the agent to process the request.

## Code Structure

### `app.py` - Main Agent Service

The main application demonstrates the full configuration pattern:

```python
# Profile configuration
profile = AgentProfileConfig(
    name="Weather Agent",
    role="Weather Assistant",
    goal="Assist with weather tasks",
    instructions=[...],
)

# Pub/Sub configuration
pubsub = AgentPubSubConfig(
    pubsub_name="messagepubsub",
    agent_topic="weather.requests",
    broadcast_topic="agents.broadcast",
)

# State, Registry, and Memory configurations
state = AgentStateConfig(...)
registry = AgentRegistryConfig(...)
memory = AgentMemoryConfig(...)

# Assemble the agent
agent = DurableAgent(
    profile=profile,
    pubsub=pubsub,
    state=state,
    registry=registry,
    memory=memory,
    llm=llm,
    tools=tools,
)

# Register pub/sub routes and wait for events
runner = AgentRunner()
runner.register_routes(agent)
await wait_for_shutdown()
```

### `message_client.py` - Event Publisher

Sends messages to trigger agent workflows via pub/sub.

### `agent_tools.py` - Tool Definitions

Contains custom tool implementations (e.g., weather lookup).

## Benefits of Using Configuration Classes

### 1. **Separation of Concerns**
Each configuration class handles a specific aspect of agent setup, making code more maintainable and testable.

### 2. **Type Safety**
Configuration classes are dataclasses with type hints, catching errors at development time.

### 3. **Reusability**
Create configuration templates and reuse them across multiple agents:

```python
# Shared config template
standard_pubsub = AgentPubSubConfig(
    pubsub_name="messagepubsub",
    broadcast_topic="agents.broadcast",
)

# Use with different agents
weather_agent = DurableAgent(pubsub=standard_pubsub, ...)
travel_agent = DurableAgent(pubsub=standard_pubsub, ...)
```

### 4. **Production-Ready**
Explicit configurations make it clear what dependencies and resources your agent requires.

### 5. **Testing & Mocking**
Easy to mock individual configurations during testing:

```python
# Test with in-memory storage
test_state = AgentStateConfig(
    store=InMemoryStateStore(),
)
```

## Learn More

- **Configuration Reference**: [`dapr_agents/agents/configs.py`](../../dapr_agents/agents/configs.py)
- **Simple Agent Example**: [`03-durable-agent-tool-call`](../03-durable-agent-tool-call/)
- **Multi-Agent Workflows**: [`05-multi-agent-workflows`](../05-multi-agent-workflows/)
- **Dapr Pub/Sub**: [Dapr Documentation](https://docs.dapr.io/developing-applications/building-blocks/pubsub/)
- **Dapr State Management**: [Dapr State Store Reference](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
