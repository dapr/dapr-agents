<!--
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 09 — Agents as Tools

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
cd examples/09-agents-as-tools
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
with `agent_to_tool`.

```bash
cd examples/09-agents-as-tools
dapr run -f dapr-cross-app.yaml
```

Trigger Frodo the same way:

```bash
curl -X POST http://localhost:8001/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What supplies do we have for the next leg of the journey?"}'
```

## How It Works

1. **Registry membership** — Sam registers itself in the shared Dapr
   state-store registry by providing a `registry=` config.
2. **Auto-discovery** — At the start of every `dapr.durableagent.frodo.workflow` run,
   `_load_tools` scans the registry and registers all peer agents (excluding
   orchestrators and self) in Frodo's `tool_executor`.
3. **LLM picks the tool** — Frodo's LLM sees `sam` as a function-call tool
   (the Dapr workflow context `ctx` is hidden from the schema).
4. **Child workflow** — The framework calls
   `ctx.call_child_workflow(workflow="dapr.durableagent.sam.workflow", input={"task": ...})`
   and yields until Sam completes.
5. **Result** — Sam's final message is returned as a `ToolMessage` and added to
   Frodo's conversation history.

## Key APIs

```python
from dapr_agents.tool.workflow.agent_tool import agent_to_tool

# Explicit cross-app factory (no registry needed)
sam_tool = agent_to_tool(
    "sam",
    description="Sam Gamgee. Goal: Manage provisions.",
    target_app_id="SamApp",
)

frodo = DurableAgent(name="frodo", tools=[sam_tool], ...)

# OR: pass a DurableAgent instance directly (auto-converted, same app)
sam = DurableAgent(name="sam", registry=registry, ...)
frodo = DurableAgent(name="frodo", tools=[sam], ...)

# OR: shared registry — all peers auto-discovered at workflow start
sam = DurableAgent(name="sam", registry=registry, ...)
frodo = DurableAgent(name="frodo", registry=registry, ...)
```
