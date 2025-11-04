# Agent Handoffs: Translation Coordinator

This quickstart demonstrates the **handoff pattern** where a coordinator agent (Translation Intake) triages incoming requests and hands off to specialized translator agents. It showcases how to use **HandoffSpec** for agent-to-agent delegation, shared conversation memory, and the fire-and-forget messaging pattern powered by Dapr pub/sub.

## What You'll Learn

- How to implement **agent handoffs** using **HandoffSpec**
- Configure a **triage agent** that routes tasks to specialized agents
- Use **shared conversation memory** across multiple agents
- Set up **agent coordination** through the registry
- Deploy a multi-agent system with **Dapr Multi-App Run**
- Understand the **fire-and-forget handoff mechanism**

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Dapr CLI and Docker
- Redis running on `localhost:6379`
- OpenAI API key (or compatible LLM provider)

## The Translation Swarm Architecture

This quickstart features a triage pattern with three specialized agents:

| Agent | Role | Handoffs | Tools | Topic |
|-------|------|----------|-------|-------|
| **Translation Intake** | Coordinator & Triage | ✅ Spanish, Italian | None | `translator.intake.requests` |
| **Spanish Translator** | Spanish translation specialist | ❌ None | None | `translator.spanish.requests` |
| **Italian Translator** | Italian translation specialist | ❌ None | None | `translator.italian.requests` |

### Key Pattern: Triage Agent

This follows a **triage pattern** where:
- ✅ Only the **intake agent** has handoff capabilities
- ❌ Specialized agents (translators) have **no handoffs**
- 🎯 Intake agent routes based on user request
- 🔄 No back-and-forth delegation between translators

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
- `translatorregistry.yaml`: Agent registry storage (shared)
- `translatormemory.yaml`: **Shared conversation memory** (Redis with `keyPrefix: none`)
- `messagepubsub.yaml`: Pub/sub messaging for handoffs

## Understanding Agent Handoffs

### What is HandoffSpec?

`HandoffSpec` defines a handoff capability that allows one agent to delegate work to another:

```python
from dapr_agents.agents import HandoffSpec

agent = DurableAgent(
    profile=...,
    pubsub=...,
    # Define handoff targets
    handoffs=[
        HandoffSpec(
            agent_name="Spanish Translator",
            description="Hand off to the Spanish translator for Spanish translation requests.",
        ),
        HandoffSpec(
            agent_name="Italian Translator",
            description="Hand off to the Italian translator for Italian translation requests.",
        ),
    ],
)
```

### How Handoffs Work

1. **LLM Decision**: The intake agent's LLM decides a handoff is needed based on instructions
2. **Synthetic Response**: `DurableAgent` generates synthetic tool and assistant responses (not actual tool execution)
3. **Message Publishing**: A message is published to the target agent's pub/sub topic
4. **Fire-and-Forget**: The handoff message is sent without waiting for a response
5. **Target Execution**: The target agent receives the message and processes it independently

### Shared Conversation Memory

**Critical Feature**: All agents share the same conversation history!

```python
# ALL agents use the SAME session_id
memory = AgentMemoryConfig(
    store=ConversationDaprStateMemory(
        store_name="translatormemorystore",
        session_id="translator.session",  # SHARED across all agents
    )
)
```

**Why This Matters:**
- The Spanish Translator can see the original user request
- The Italian Translator can see the entire conversation
- Each agent adds to the shared conversation thread
- Creates a unified transcript across all handoffs

**Memory Store Configuration:**
```yaml
# components/translatormemory.yaml
metadata:
  - name: keyPrefix
    value: "none"  # Allows cross-agent memory sharing
```

## Running the Translation Swarm

### Start All Services

```bash
dapr run -f dapr-swarm.yaml
```

**What's running:**
- ✅ Translation Intake (coordinator)
- ✅ Spanish Translator
- ✅ Italian Translator
- ✅ Client (publishes translation request)

### Expected Flow

1. **Client** → publishes task to `translator.intake.requests`
   ```
   "Translate 'Hello, how are you?' to Spanish"
   ```

2. **Translation Intake** → receives message, LLM analyzes request
   - Identifies: Spanish translation needed
   - Decides: Hand off to Spanish Translator
   - Generates synthetic handoff responses
   - Publishes to `translator.spanish.requests`

3. **Spanish Translator** → receives handoff message
   - Reads shared conversation history (sees original request)
   - Performs translation
   - Returns result: "Hola, ¿cómo estás?"
   - Writes to shared conversation memory

### Viewing the Shared Conversation

Because all agents write to the same `translator.session` in Redis:

```bash
# View the complete conversation thread
redis-cli GET translator.session
```

You'll see:
- Original user request
- Intake agent's analysis
- Synthetic handoff tool call
- Translator's response

All in one unified conversation!

## Manual Testing

### Start Agents Separately

```bash
# Terminal 1: Translation Intake
cd services/translator
dapr run --app-id translator-intake --resources-path ../../components -- python3 app.py

# Terminal 2: Spanish Translator
cd services/spanish
dapr run --app-id spanish-translator --resources-path ../../components -- python3 app.py

# Terminal 3: Italian Translator
cd services/italian
dapr run --app-id italian-translator --resources-path ../../components -- python3 app.py
```

### Send Test Requests

```bash
cd services/client

# Test Spanish translation
python3 publish_task.py "Translate 'Good morning' to Spanish"

# Test Italian translation  
python3 publish_task.py "Translate 'Thank you very much' to Italian"

# Test with context
python3 publish_task.py "I need a formal Spanish translation of 'Welcome to our company'"
```

## Project Structure

```
10-handoffs-translator/
├── components/                    # Dapr components
│   ├── messagepubsub.yaml        # Pub/sub for handoffs
│   ├── openai.yaml               # LLM provider
│   ├── translatorregistry.yaml   # Shared registry
│   ├── translatormemory.yaml     # Shared memory (keyPrefix: none)
│   └── workflowstate.yaml        # Workflow state
├── services/
│   ├── translator/               # Intake/coordinator agent
│   │   └── app.py
│   ├── spanish/                  # Spanish translator agent
│   │   └── app.py
│   ├── italian/                  # Italian translator agent
│   │   └── app.py
│   └── client/                   # Test client
│       └── publish_task.py
├── dapr-swarm.yaml               # Multi-app run config
└── README.md
```

## Key Concepts Demonstrated

### 1. Triage Pattern
- One coordinator agent with handoffs
- Multiple specialized agents without handoffs
- Simple one-way delegation

### 2. Shared Memory
- All agents write to `translator.session`
- Redis configured with `keyPrefix: none`
- Unified conversation transcript

### 3. Fire-and-Forget Handoffs
- Intake agent doesn't wait for translator response
- Asynchronous message-based communication
- Decoupled agent execution

### 4. Agent Registry
- All agents register in `translator-swarm` team
- Enables agent discovery
- Shared coordination metadata

## Troubleshooting

### Issue: Translator agents not receiving handoffs

**Solution:** Verify topics match between handoff and agent configuration:
```python
# In intake agent HandoffSpec
agent_name="Spanish Translator"

# In Spanish translator pubsub config
agent_topic="translator.spanish.requests"
```

The agent name is used to lookup the registered agent's topic!

### Issue: Translator can't see original request

**Solution:** Ensure all agents use the same `session_id`:
```python
session_id="translator.session"  # Must be identical
```

### Issue: Memory not persisting

**Solution:** Check Redis is running and accessible:
```bash
redis-cli PING
# Should return: PONG
```

## Clean Up

Stop all services:
```bash
# Press Ctrl+C in the dapr run terminal
```

Clear shared conversation history:
```bash
redis-cli FLUSHDB
```

## Learn More

- **Configuration Classes**: [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/) - Complete config guide
- **Multi-Agent Workflows**: [`05-multi-agent-workflows-new`](../05-multi-agent-workflows-new/) - Orchestration patterns
- **Advanced Handoffs**: [`11-handoffs-research-writer`](../11-handoffs-research-writer/) - Cyclic handoff workflow
- **Dapr Pub/Sub**: [Pub/Sub Building Block](https://docs.dapr.io/developing-applications/building-blocks/pubsub/)


## Learn More

- **Configuration Classes**: [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/) - Complete config guide
- **Multi-Agent Workflows**: [`05-multi-agent-workflows-new`](../05-multi-agent-workflows-new/) - Orchestration patterns
- **Advanced Handoffs**: [`11-handoffs-research-writer`](../11-handoffs-research-writer/) - Cyclic handoff workflow
- **Dapr Pub/Sub**: [Pub/Sub Building Block](https://docs.dapr.io/developing-applications/building-blocks/pubsub/)

1. **Client** → publishes task to `translator.intake.requests` topic
2. **Translation Intake Agent** → receives message, LLM suggests handoff
3. **DurableAgent Logic** → generates synthetic tool/assistant responses, publishes to translator's topic
4. **Spanish/Italian Translator** → receives message, sees full conversation history in shared memory, performs translation
5. **Result** → translator returns final translation directly

## Clean up

Stop the Dapr run with `Ctrl+C`. 

To clear the shared conversation history and start fresh:

```bash
redis-cli FLUSHDB
```

This removes all keys from the current Redis database, including the `translator.session` conversation memory shared by all agents.
