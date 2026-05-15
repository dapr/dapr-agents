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

# Echo Agent Executor

End-to-end validation for the `AgentExecutorBase` abstraction introduced in
RFC [dapr/dapr-agents#569](https://github.com/dapr/dapr-agents/issues/569).
It runs a `DurableAgent` backed by `EchoAgentExecutor` — a trivial,
zero-dependency executor that echoes the prompt back through the full event
stream (`text_delta` → `message` → `session` → `complete`).

No LLM provider or API key is required.

## What it verifies

* `DurableAgent(executor=...)` constructs without an `llm` and auto-wires
  Dapr infrastructure (state store, pub/sub, workflow state) through the
  standard `AgentRunner` path.
* `agent_workflow` takes the new `elif self.executor is not None:` branch
  and dispatches the `run_executor` activity.
* `_consume_executor` drives the executor's async event stream and
  persists the user + assistant messages into
  `AgentWorkflowEntry.messages`, with `session_id` populated, via the
  existing `DaprInfra.save_state` path.
* `EchoAgentExecutor.get_session(...)` returns the in-memory transcript
  after the run completes.

## Prerequisites

* Python ≥ 3.11
* [`uv`](https://docs.astral.sh/uv/)
* Dapr CLI + runtime (`dapr init`)
* Redis reachable on `localhost:6379` (the default `dapr init` sidecar
  provides this; `docker run -p 6379:6379 redis:7` also works)

## Setup

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync --active
```

## Run

From this directory:

```bash
dapr run \
  --app-id echo-executor-app \
  --resources-path ./resources \
  -- \
  python app.py
```

Expected output (trimmed):

```
=== Final Result ===
{'role': 'assistant', 'content': 'echo: hello, agent executor', ...}
====================
Inspect state: `redis-cli --scan --pattern 'echo*'` then `redis-cli GET <key>` …
```

## Inspecting Dapr state

The run persists an `AgentWorkflowEntry` under the key
`echo:_workflow_<workflow-instance-id>` in the `agentstatestore` Redis
component. To confirm the executor branch wrote the session:

```bash
redis-cli --scan --pattern 'echo*' | head
redis-cli GET <one-of-those-keys>
```

The JSON should include a populated `session_id` field and a `messages`
array with one user + one assistant entry. The `tool_history` array will
be empty (the echo executor emits no `tool_call` events).

## Teardown

```bash
dapr stop --app-id echo-executor-app
```

## Related

* RFC: [dapr/dapr-agents#569](https://github.com/dapr/dapr-agents/issues/569)
* Linear: AI‑505 (this ticket), AI‑506 (Claude Agent SDK provider —
  the first *real* `AgentExecutorBase`), AI‑509 (end‑to‑end pipeline‑stage
  validation).
