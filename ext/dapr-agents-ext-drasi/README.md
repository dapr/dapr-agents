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
    drasi_trigger,                  # Register Drasi query subscriptions for agents
    DrasiChangeEvent,               # Type for Drasi change events
    DrasiOperation,                 # Operation type for Drasi change events
)
```

### Usage

Register a Drasi query subscription on an agent before hosting:

```python
agent = DurableAgent(...)

drasi_trigger(
    agent,
    query_id="<YOUR_DRASI_QUERY_ID>",
    task_mapper=lambda event, ctx: TriggerAction(task="<AGENT_TASK_MESSAGE>")
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

### Regenerate Drasi models

See the [provenance file](./PROVENANCE.md) for context.

```bash
./scripts/regen-drasi-models.sh
```