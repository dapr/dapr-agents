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

# dapr-agents-ext-drasi

[Drasi](https://drasi.io/) extension for Dapr Agents, enabling resilient, scalable, and business event-driven AI agents through Drasi's change notification capabilities.

## Features

- Directly trigger agents from Drasi change events via Dapr pub/sub

## Getting Started

### Prerequisites

This extension is installed as an optional dependency on the core `dapr-agents` package; see the `Getting Started` section in the [root README](../../README.md) for a list of prerequisites.

### Installation

```bash
uv add dapr-agents[drasi]
```

### Public API

```python
from dapr_agents.ext.drasi import (
    drasi_trigger,  # Register Drasi query subscriptions for agents
    DrasiChangeEvent,  # Type for Drasi change events
    DrasiOperation,  # Operation type for Drasi change events
)
```

### Usage

Register a Drasi query subscription on an agent before hosting:

```python
agent = DurableAgent(...)

drasi_trigger(
    agent,
    query_id="<YOUR_DRASI_QUERY_ID>",
    task_mapper=lambda event, ctx: TriggerAction(task="<AGENT_TASK_MESSAGE>"),
)

runner = AgentRunner()
try:
    runner.subscribe(agent)
    await wait_for_shutdown()
finally:
    runner.shutdown(agent)
```

### Examples
- [Drasi Change-Driven Agents on Kubernetes](../../examples/ext-drasi-change-driven-agents-k8s/README.md) — demonstrates how to subscribe an agent to Drasi queries in a Kubernetes environment.

## Development

### Install extension in editable mode

From the project root:

```bash
uv venv
source .venv/bin/activate
uv sync --active --group dev --group test --extra drasi
```

### Run extension tests

```bash
uv run --group test pytest ext/dapr-agents-ext-drasi -m "not integration" -v
```

### Extension code quality

See the `Code Quality` section in the [development README](../../docs/development/README.md) for code quality commands.

### Private subscription components

The subscription modules are internal groundwork, not a public agent lifecycle.
They do not change the exports above or attach tools, host agents, consume inbox
events, or schedule workflows. Importing them does not start runtime resources.
The extension alone depends on `drasi-agent-router-contracts==0.1.0a1`; its
published models, validation, identity helpers, and packaged fixtures are reused
without a Drasi Platform checkout or code generation.

`_models.py` and `_interfaces.py` define the scope, durable intent, outcomes, and
component boundaries. `intent_store.py` adapts the configured agent state-store
primitive. `router_client.py` validates scope-bound MCP exchanges through Dapr
service invocation. `subscription_manager.py` manages transitions and recovery.
`subscription_tools.py` constructs ordinary `AgentTool` instances without I/O or
attachment: query IDs are bound in code, while the model supplies only operation
filters and self-contained instructions. Instructions remain in local intent,
never in router requests. Listing reports local intent, not live router health.

Preparation requires one active owner for the router/subscriber scope. That owner
loads the scoped document, initializes it only after establishing absence, reloads
its ETag, and reconciles against the validated operator catalog before admitting
commands. Initialization is not atomic create-if-absent. An empty catalog is valid;
a failed catalog read is not an empty catalog. The catalog is a fixed, curated
snapshot, not a query-creation or live-refresh API.

The durable transitions are:

```text
absent -> pending_subscribe -> active
active -> pending_update -> active
active/unavailable -> pending_unsubscribe -> absent intent
retired query -> unavailable (historical intent retained)
```

Pending intent is committed before router I/O; success follows the final local
commit. Rejections and uncertain outcomes remain explicit. Timeouts or malformed
confirmations can follow a committed remote mutation, so they retain pending
intent rather than implying rollback. Reconciliation finishes pending removals
first, reasserts supported intent without invoking a model, and retains unavailable
intent without automatic revival. Updates and recovery preserve incarnation;
a completed unsubscribe followed by a new subscription creates a new one.

All queries share one document and one ETag. Conflicts reload the document and
merge only the targeted query; a competing same-query transition is not overwritten.
Reads return owned snapshots. The adapter narrowly accesses the configured store's
SDK client because the read wrapper does not expose strong consistency and the
write wrapper cannot disable automatic retries per call. Writes use first-write
concurrency and no SDK retries; strong reads retain the configured retry policy.
The adapter does not change the agent's store configuration, workflow model, or
local mirroring behavior. Tests exercise native SDK writes and stale-replica reads.

The owner closes the router client, never the borrowed agent state store.
Each bounded MCP exchange closes its own stateless resources; closing the client
does not delete durable routing rules. State/network methods are synchronous:
async callers must offload them, and deterministic workflow bodies must not call
them. There is no distributed fencing, guaranteed replay, or scale-to-zero claim.

Run extension and core tests in separate processes, using the existing import guard:

```bash
uv sync --group test --extra drasi
DAPR_AGENTS_REQUIRE_DRASI=1 uv run pytest ext -m "not integration"
uv run pytest tests -m "not integration"
```

The six `test_subscription_*.py`, `test_intent_store.py`, and
`test_router_client.py` modules cover this private boundary with controlled
transport and storage fixtures. No model credentials or cluster are needed.
Admission decisions, delivery/scheduling, public lifecycle hosting, and runtime
examples (including their tests) are deferred to later contributions. External
user documentation and release metadata for the usable public feature belong
with that lifecycle, not this private step.

### Regenerate Drasi models

See the [provenance file](./PROVENANCE.md) for context.

```bash
./scripts/regen-drasi-models.sh
```