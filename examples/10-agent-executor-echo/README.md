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

A self-contained example showing how to run a `DurableAgent` with a
stateful agent runtime instead of a chat-completion LLM client. The agent
is wired up with `EchoAgentExecutor` — a trivial, zero-dependency executor
that echoes the prompt back over the streaming event protocol — so you can
see the full executor flow without provisioning an LLM provider or API key.

## What you'll see

* Build a `DurableAgent` by passing `executor=...` instead of `llm=...`
  and run it through the standard `AgentRunner` interface — no extra
  glue code required.
* Watch the agent receive the user prompt and stream back an echoed
  assistant response, end-to-end, through Dapr workflows + pub/sub +
  state store.
* Inspect persisted Dapr state to see the user and assistant messages
  recorded against a populated `session_id`, ready to resume a follow-up
  turn from where the previous one left off.

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
