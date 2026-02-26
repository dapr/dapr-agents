# 05 — Agents as Tools

This example demonstrates how one `DurableAgent` can call another as a
**synchronous child workflow tool**.  The calling agent's LLM picks the tool,
the framework schedules the target agent's workflow as a Dapr child workflow,
and the result flows back as a `ToolMessage` — all without leaving the durable
workflow execution model.

Two scenarios are provided:

| Scenario | Description |
|---|---|
| `in-process` | Sam and Frodo run in the **same Dapr app** / Python process. |
| `cross-app` | Sam and Frodo run in **separate Dapr apps** and communicate via Dapr multi-app child workflow routing. |

## Prerequisites

- Dapr CLI installed and a local Redis instance running
- Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- `OPENAI_API_KEY` environment variable set (or configure a different LLM provider in `components/llm-provider.yaml`)

## In-Process Scenario

Both agents share one Dapr sidecar.  Sam is passed directly as a `DurableAgent`
instance in Frodo's `tools` list and is auto-converted to an `AgentWorkflowTool`.

```bash
cd examples/05-agents-as-tools
dapr run -f dapr-in-process.yaml
```

Trigger Frodo:

```bash
curl -X POST http://localhost:8001/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "How many lembas breads do we have left? Ask Sam."}'
```

## Cross-App Scenario

Sam and Frodo each have their own Dapr sidecar.  Frodo discovers Sam via the
shared registry (`USE_REGISTRY=1`, the default) or you can wire them explicitly
with `convert_agent_as_tool`.

```bash
cd examples/05-agents-as-tools
dapr run -f dapr-cross-app.yaml
```

Trigger Frodo the same way:

```bash
curl -X POST http://localhost:8001/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What supplies do we have for the next leg of the journey?"}'
```

## How It Works

1. **`is_tool=True`** — Sam registers itself in the shared Dapr state-store
   registry with `agent.is_tool = true`.
2. **Auto-discovery** — At the start of every `frodo_agent_workflow` run,
   `_load_tool_agents` scans the registry for agents with `is_tool=True` and
   registers them in Frodo's `tool_executor`.
3. **LLM picks the tool** — Frodo's LLM sees `sam` as a function-call tool
   (the Dapr workflow context `ctx` is hidden from the schema).
4. **Child workflow** — The framework calls
   `ctx.call_child_workflow(workflow="sam_agent_workflow", input={"task": ...})`
   and yields until Sam completes.
5. **Result** — Sam's final message is returned as a `ToolMessage` and added to
   Frodo's conversation history.

## Key APIs

```python
from dapr_agents.tool.workflow import convert_agent_as_tool

# Explicit cross-app factory (no registry needed)
sam_tool = convert_agent_as_tool(
    "sam",
    description="Sam Gamgee. Goal: Manage provisions.",
    target_app_id="SamApp",
)

frodo = DurableAgent(name="frodo", tools=[sam_tool], ...)

# OR: pass a DurableAgent instance (auto-converted, same app)
sam = DurableAgent(name="sam", is_tool=True, ...)
frodo = DurableAgent(name="frodo", tools=[sam], ...)

# OR: pass a string name (deferred registry lookup)
frodo = DurableAgent(name="frodo", tools=["sam"], registry=registry, ...)
```
