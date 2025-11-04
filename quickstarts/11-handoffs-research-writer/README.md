# Agent Handoffs: Research & Writing Workflow

This quickstart demonstrates a **cyclic multi-agent workflow** where research, writing, and reviewing agents collaborate through bidirectional handoffs. Unlike the simple triage pattern, this showcases a complex workflow where agents can hand off to multiple targets, creating revision loops until the work meets quality standards. This example is inspired by LlamaIndex agent workflows but implemented using Dapr Agents.

## What You'll Learn

- Implement **bidirectional handoffs** between multiple agents
- Create **cyclic workflows** with revision loops (write → review → revise → review)
- Use **agent tools** in combination with handoffs
- Configure **shared conversation memory** for workflow context
- Build **complex multi-step workflows** with agent coordination
- Handle **iterative refinement** patterns in agent systems

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- Dapr CLI and Docker
- Redis running on `localhost:6379`
- OpenAI API key (or compatible LLM provider)

## The Research Workflow Architecture

This quickstart features a cyclic workflow with three collaborative agents:

| Agent | Role | Tools | Handoffs | Topic |
|-------|------|-------|----------|-------|
| **Research Agent** | Gathers information | `search_web`, `record_notes` | ➡️ Writer Agent | `research.requests` |
| **Writer Agent** | Creates reports | `write_report` | ➡️ Reviewer, Research | `writer.requests` |
| **Reviewer Agent** | Provides feedback | `review_report` | ➡️ Writer Agent | `reviewer.requests` |

### Workflow Pattern: Cyclic Collaboration

```
User Request → Research Agent → Writer Agent → Reviewer Agent
                     ↑               ↓              ↓
                     └───────────────┴──────────────┘
                   (revision loop until approved)
```

**Key Differences from Translator Example:**
- ✅ Multiple agents have handoff capabilities (not just one coordinator)
- ✅ Bidirectional handoffs (Writer can go to both Reviewer AND Research)
- ✅ Cyclic workflow (can loop multiple times)
- ✅ Each agent has specialized tools
- ✅ Complex state management across iterations

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
- `researchregistry.yaml`: Agent registry storage (shared across all agents)
- `researchmemory.yaml`: **Shared conversation memory** (Redis with `keyPrefix: none`)
- `messagepubsub.yaml`: Pub/sub messaging for handoffs

## Shared Memory for Workflow Context

**Critical for Cyclic Workflows**: All agents must see the entire conversation history!

```python
# ALL agents use the SAME session_id
memory = AgentMemoryConfig(
    store=ConversationDaprStateMemory(
        store_name="researchmemorystore",
        session_id="research.session",  # SHARED across all agents
    )
)
```

**Why This is Essential:**
- Writer Agent sees Research Agent's notes
- Reviewer Agent sees the written report
- Writer Agent sees Reviewer's feedback when revising
- Creates a complete audit trail of the workflow

**Memory Store Configuration:**
```yaml
# components/researchmemory.yaml
metadata:
  - name: keyPrefix
    value: "none"  # Enables cross-agent memory access
```

## Agent Tools (Simulated)

This quickstart uses **fake tools** to demonstrate the workflow pattern without external dependencies:

### Research Tools (`services/research_tools.py`)

```python
@tool(args_model=SearchWebSchema)
def search_web(query: str) -> str:
    """Search the web for information (simulated)."""
    # Returns pre-defined search results based on keywords
    # In production: would call real search API

@tool(args_model=RecordNotesSchema)
def record_notes(title: str, notes: str) -> str:
    """Record research notes (simulated)."""
    # Acknowledges note recording
    # In production: would save to database
```

### Writer Tools

```python
@tool(args_model=WriteReportSchema)
def write_report(title: str, content: str) -> str:
    """Write a report (simulated)."""
    # Acknowledges report creation
    # In production: would save document to storage
```

### Reviewer Tools

```python
@tool(args_model=ReviewReportSchema)
def review_report(feedback: str, approved: bool) -> str:
    """Review a report (simulated)."""
    # Acknowledges review submission
    # In production: would save feedback and status
```

## Running the Research Workflow

### Start All Services

```bash
dapr run -f dapr-swarm.yaml
```

**What's running:**
- ✅ Research Agent
- ✅ Writer Agent
- ✅ Reviewer Agent
- ✅ Client (publishes research request)

### Expected Workflow Flow

1. **Client** → publishes task to `research.requests`
   ```
   "Research and write a report about artificial intelligence in healthcare"
   ```

2. **Research Agent** → receives message
   - Uses `search_web("AI healthcare applications")`
   - Uses `search_web("AI medical diagnosis")`
   - Uses `record_notes("AI in Healthcare", "Key findings: ...")`
   - Hands off to **Writer Agent**

3. **Writer Agent** → receives handoff
   - Reviews research notes from shared memory
   - Uses `write_report("AI in Healthcare Report", "# Introduction\n...")`
   - Hands off to **Reviewer Agent**

4. **Reviewer Agent** → receives handoff
   - Reviews report from shared memory
   - Uses `review_report("Good structure, but needs more details on risks", approved=False)`
   - Hands off back to **Writer Agent**

5. **Writer Agent** → receives feedback handoff
   - Reviews feedback from shared memory
   - Revises report with more details
   - Hands off to **Reviewer Agent** again

6. **Reviewer Agent** → receives revised report
   - Reviews improvements
   - Uses `review_report("Excellent report, approved", approved=True)`
   - Workflow completes ✅

## Project Structure

```
11-handoffs-research-writer/
├── components/                    # Dapr components
│   ├── messagepubsub.yaml        # Pub/sub for handoffs
│   ├── openai.yaml               # LLM provider
│   ├── researchregistry.yaml     # Shared agent registry
│   ├── researchmemory.yaml       # Shared memory (keyPrefix: none)
│   └── workflowstate.yaml        # Workflow state
├── services/
│   ├── research/                 # Research agent
│   │   └── app.py
│   ├── writer/                   # Writer agent
│   │   └── app.py
│   ├── reviewer/                 # Reviewer agent
│   │   └── app.py
│   ├── research_tools.py         # Shared tool implementations
│   └── client/                   # Test client
│       └── publish_task.py
├── dapr-swarm.yaml               # Multi-app run config
└── README.md
```

## Key Concepts Demonstrated

### 1. Cyclic Workflow Pattern
- Multiple agents with handoff capabilities
- Bidirectional handoffs create revision loops
- Complex state management across iterations

### 2. Tool + Handoff Combination
- Each agent has specialized tools for their domain
- Tools execute before handoffs
- Tools provide work artifacts visible to next agent

### 3. Shared Workflow Context
- All agents access the same conversation thread
- Enables context awareness across handoffs
- Creates complete workflow audit trail

### 4. Iterative Refinement
- Reviewer can send work back for revisions
- Multiple review cycles until approval
- Quality gates enforced through handoff logic

## Comparison with Translator Example

| Feature | Translator (Simple) | Research-Writer (Complex) |
|---------|-------------------|--------------------------|
| Handoff Pattern | One-way triage | Bidirectional cyclic |
| Coordinator | Single (Intake) | Multiple (all agents) |
| Workflow Type | Linear | Cyclic with loops |
| Tools | None | Each agent has tools |
| Iterations | Single pass | Multiple revision cycles |
| Complexity | Low | High |
| Use Case | Simple routing | Complex workflows |

## Troubleshooting

### Issue: Agents stuck in infinite loop

**Solution:** Add iteration limits in agent instructions:
```python
instructions=[
    "After 2 revision cycles, accept the report even if minor issues remain.",
]
```

### Issue: Agents can't see previous work

**Solution:** Verify all agents use identical session ID:
```python
session_id="research.session"  # Must match across ALL agents
```

### Issue: Handoffs not reaching target agent

**Solution:** Check agent name in HandoffSpec matches registered agent:
```python
# HandoffSpec uses agent name to lookup topic
HandoffSpec(agent_name="Writer Agent")  # Must match agent's profile.name
```

### Issue: Tools not executing

**Solution:** Verify tools are passed to agent and imported correctly:
```python
from research_tools import research_tools

agent = DurableAgent(
    # ...
    tools=research_tools,  # Don't forget to pass tools!
)
```

## Learn More

- **Simple Handoffs**: [`10-handoffs-translator`](../10-handoffs-translator/) - Triage pattern introduction
- **Configuration Classes**: [`03-durable-agent-with-configs`](../03-durable-agent-with-configs/) - Complete config guide
- **Multi-Agent Workflows**: [`05-multi-agent-workflows-new`](../05-multi-agent-workflows-new/) - Orchestration patterns
- **Agent Tools**: [`03-agent-tool-call`](../03-agent-tool-call/) - Tool usage patterns
- **Dapr Pub/Sub**: [Pub/Sub Building Block](https://docs.dapr.io/developing-applications/building-blocks/pubsub/)
