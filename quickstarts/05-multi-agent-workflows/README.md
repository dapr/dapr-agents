# Multi-Agent Workflows with Configuration Classes

This quickstart demonstrates how to create and orchestrate event-driven workflows with multiple autonomous agents using **Agent Configuration Classes** from Dapr Agents. You'll learn how to set up agents as separate services, implement three different workflow orchestration patterns, and enable real-time agent collaboration through pub/sub messaging—all using the modern configuration-based approach.

## What You'll Learn

- How to use **Agent Configuration Classes** for production-ready multi-agent systems
- Configure multiple agents with **AgentProfileConfig**, **AgentPubSubConfig**, **AgentStateConfig**, **AgentRegistryConfig**, and **AgentMemoryConfig**
- Implement three orchestration patterns: **Random**, **Round-Robin**, and **LLM-based**
- Deploy agents as separate microservices using **Dapr Multi-App Run**
- Enable agent collaboration through pub/sub messaging and shared registries
- Send messages to orchestrators or directly to individual agents

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Dapr CLI and Docker
- OpenAI API key (or compatible LLM provider)

## The Fellowship: Multi-Agent Architecture

This quickstart features four agents from the Fellowship of the Ring, each with distinct roles and expertise:

| Agent | Role | Expertise | Topic |
|-------|------|-----------|-------|
| **Frodo** | Hobbit & Ring-bearer | Endurance, stealth, burden-bearing | `fellowship.frodo.requests` |
| **Sam** | Logistics & Support | Supplies, provisions, morale | `fellowship.sam.requests` |
| **Gandalf** | Wizard & Loremaster | Strategy, magic, ancient knowledge | `fellowship.gandalf.requests` |
| **Legolas** | Elf Scout & Marksman | Scouting, archery, terrain navigation | `fellowship.legolas.requests` |

### Orchestration Patterns

Three orchestrators coordinate the Fellowship:

1. **Random Orchestrator** (`fellowship.orchestrator.random.requests`)
   - Randomly selects an agent for each task
   - Good for load distribution and testing

2. **Round-Robin Orchestrator** (`fellowship.orchestrator.roundrobin.requests`)
   - Cycles through agents sequentially
   - Ensures fair task distribution

3. **LLM Orchestrator** (`llm.orchestrator.requests`)
   - AI-powered agent selection based on task content
   - Intelligent routing to the most appropriate agent

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

### Option 1: Using Environment Variables (Recommended)

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
```

2. Export environment variables before running:
```bash
# Get environment variables from .env file
export $(grep -v '^#' ../../.env | xargs)
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

Make sure Dapr is initialized:

```bash
dapr init
```

This quickstart uses these Dapr components (in `components/` directory):
- `openai.yaml`: LLM conversation component
- `workflowstate.yaml`: Workflow state storage
- `memorystore.yaml`: Conversation memory storage
- `agentregistrystore.yaml`: Agent registry storage (shared across all agents)
- `messagepubsub.yaml`: Pub/sub messaging for agent communication

## Project Structure

```
05-multi-agent-workflows-new/
├── components/                    # Dapr configuration files
│   ├── agentregistrystore.yaml    # Shared agent registry
│   ├── memorystore.yaml           # Conversation memory
│   ├── messagepubsub.yaml         # Pub/sub messaging
│   ├── openai.yaml                # LLM provider
│   └── workflowstate.yaml         # Workflow state
├── services/                      # Agent and orchestrator services
│   ├── frodo/                     # Frodo agent service
│   │   └── app.py
│   ├── sam/                       # Sam agent service
│   │   └── app.py
│   ├── gandalf/                   # Gandalf agent service
│   │   └── app.py
│   ├── legolas/                   # Legolas agent service
│   │   └── app.py
│   ├── workflow-random/           # Random orchestrator
│   │   └── app.py
│   ├── workflow-roundrobin/       # Round-robin orchestrator
│   │   └── app.py
│   ├── workflow-llm/              # LLM-based orchestrator
│   │   └── app.py
│   └── client/                    # Client for publishing messages
│       └── pubsub_client.py
├── dapr-random.yaml               # Multi-app config: Random orchestrator
├── dapr-roundrobin.yaml           # Multi-app config: Round-robin orchestrator
├── dapr-llm.yaml                  # Multi-app config: LLM orchestrator
└── README.md
```

## Agent Configuration with Config Classes

Each agent in this quickstart uses the latest **Agent Configuration Classes** updates.

### Key Benefits of Configuration Classes

1. **Shared Registry**: All agents use the same `team_name="fellowship"`, enabling discovery and coordination
2. **Isolated State**: Each agent has its own state store with unique key prefix
3. **Individual Topics**: Each agent subscribes to its own direct message topic
4. **Broadcast Channel**: All agents can receive team-wide broadcasts via `fellowship.broadcast`
5. **Type Safety**: Configuration classes provide validation and type checking

## Running the Multi-Agent System

The quickstart includes three Dapr Multi-App Run configurations, each showcasing a different orchestration pattern.

### Option 1: Random Orchestrator

Randomly selects agents for each task:

```bash
dapr run -f dapr-random.yaml
```

**What's running:**
- ✅ Frodo agent
- ✅ Sam agent  
- ✅ Random orchestrator
- ✅ Client (publishes to `fellowship.orchestrator.random.requests`)

### Option 2: Round-Robin Orchestrator

Cycles through agents sequentially:

```bash
dapr run -f dapr-roundrobin.yaml
```

**What's running:**
- ✅ Frodo agent
- ✅ Sam agent
- ✅ Gandalf agent
- ✅ Round-robin orchestrator
- ✅ Client (publishes to `fellowship.orchestrator.roundrobin.requests`)

### Option 3: LLM Orchestrator

AI-powered agent selection based on task:

```bash
dapr run -f dapr-llm.yaml
```

**What's running:**
- ✅ Frodo agent
- ✅ Sam agent
- ✅ Gandalf agent
- ✅ LLM orchestrator
- ✅ HTTP client (hits the orchestrator’s `/run` endpoint)

### Triggering Workflows

Each `dapr-*.yaml` spins up an `HttpClientApp` that keeps POSTing to `/run` until the orchestrator responds (max 10 tries with a 5s delay). You’ll see the returned instance ID printed in that client’s console.

If you prefer to drive the system via pub/sub (for multiple tasks, or to target a specific orchestrator), run:

```bash
python services/client/pubsub_client.py --orchestrator llm
python services/client/pubsub_client.py --orchestrator random
python services/client/pubsub_client.py --orchestrator roundrobin
```

That publishes `TriggerAction` messages onto the same topics the orchestrators subscribe to.


### Expected Output

You'll see logs from all services showing:
1. **Agent startup**: Each agent registers with the fellowship registry
2. **Client message**: Task published to orchestrator topic
3. **Orchestration**: Orchestrator selects and routes to an agent
4. **Agent response**: Selected agent processes the task
5. **Workflow completion**: Final result logged

## How Agent Coordination Works

### 1. Shared Registry
All agents register themselves in the fellowship registry:
```python
registry = AgentRegistryConfig(
    store=StateStoreService(store_name="agentregistrystore"),
    team_name="fellowship",  # SAME for all agents
)
```

### 2. Message Flow

**Via Orchestrator:**
```
Client → Orchestrator Topic → Orchestrator Logic → Agent Topic → Specific Agent
```

**Direct to Agent:**
```
Client → Agent Topic → Specific Agent
```

### 3. Agent Discovery

Orchestrators query the registry to discover available agents:
```python
# Orchestrator can find all fellowship members
available_agents = registry.get_team_members("fellowship")
# Returns: [frodo, sam, gandalf, legolas]
```

### 4. Broadcast Messages

Send to all agents simultaneously:
```python
# All agents subscribed to fellowship.broadcast receive this
pubsub_client.py --topic fellowship.broadcast --task "Emergency: Nazgûl approaching!"
```

## Agent Profiles Summary

### Frodo - Ring-bearer
- **Focus**: Endurance, stealth, burden management
- **Style**: Humble, determined, shows vulnerability but maintains courage
- **Best for**: Tasks requiring persistence and careful navigation

### Sam - Logistics Expert
- **Focus**: Supplies, provisions, morale, practical support
- **Style**: Warm, plain-spoken, grounded, loyal
- **Best for**: Resource management and practical problem-solving

### Gandalf - Wizard & Strategist
- **Focus**: Strategy, lore, magic, long-term planning
- **Style**: Wise, patient, mysterious yet clear in critical moments
- **Best for**: Complex decisions requiring wisdom and foresight

### Legolas - Scout & Marksman
- **Focus**: Scouting, threat detection, terrain navigation, ranged tactics
- **Style**: Graceful, precise, observant, elvish elegance
- **Best for**: Reconnaissance and tactical positioning

## Troubleshooting

### Issue: Agents not discovering each other

**Solution:** Verify all agents use the same registry configuration:
```python
registry = AgentRegistryConfig(
    store=StateStoreService(store_name="agentregistrystore"),
    team_name="fellowship",  # Must be identical
)
```

### Issue: Messages not reaching agents

**Solution:** Check topic names match between client and agent configurations:
- Client publishes to: `fellowship.frodo.requests`
- Agent subscribes to: `fellowship.frodo.requests` (must match exactly)

### Issue: Orchestrator not finding agents

**Solution:** Ensure orchestrator has access to the registry and agents are started first.

## Learn More

- **Configuration Classes Guide**: [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/) - Complete guide to all configuration classes
- **Simple Agent Example**: [`03-durable-agent-tool-call`](../03-durable-agent-tool-call/) - Basic single-agent pattern
- **Message Router**: [`04-message-router-workflow`](../04-message-router-workflow/) - Pub/sub workflow patterns
- **Dapr Multi-App Run**: [Dapr Documentation](https://docs.dapr.io/developing-applications/local-development/multi-app-dapr-run/)
- **Dapr Pub/Sub**: [Pub/Sub Building Block](https://docs.dapr.io/developing-applications/building-blocks/pubsub/)
