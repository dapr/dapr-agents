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

# Claude Agent Executor

This example runs a `DurableAgent` whose reasoning loop is the
[Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
instead of a chat-completion LLM client. Dapr Workflows still own
durability: every run is a workflow, the Claude session transcript is kept
in a Dapr state store, and a sensitive tool call waits for a human decision
without holding a process open.

It builds on the [echo executor example](../10-agent-executor-echo), which
shows the executor wiring with no LLM at all.

## What you'll see

* **Claude as the agent runtime.** The agent is built with
  `executor=ClaudeAgentExecutor(...)` instead of `llm=...` and runs through
  the normal `AgentRunner`.
* **A dapr-agents tool exposed to Claude.** `get_weather` is an ordinary
  `@tool` function. Claude sees it as `mcp__dapr__get_weather` and it runs
  inside the agent's workflow activity.
* **Session resume.** The second workflow run passes the same `session_id`.
  The executor loads the Claude transcript from the Dapr state store and
  Claude remembers the first turn. This also works after a restart or on a
  different host.
* **Durable human approval.** A `before_tool_call` hook returns
  `RequireApproval` for `transfer_money`. Claude's run pauses, the workflow
  publishes an approval request and waits for an event, and the Claude
  session resumes with the decision once it arrives.

## Prerequisites

* Python 3.11 to 3.13
* [`uv`](https://docs.astral.sh/uv/) (or `pip`)
* Dapr CLI and runtime, initialized with `dapr init`
* Redis on `localhost:6379` (`dapr init` starts one)
* Claude credentials, one of:
  * an [Anthropic API key](https://console.anthropic.com/) in
    `ANTHROPIC_API_KEY`, or
  * for local development only, a logged-in
    [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) CLI
    (`claude` then `/login`). With `ANTHROPIC_API_KEY` unset, the SDK uses
    that login.

### About the `claude` extra

`ClaudeAgentExecutor` needs the optional `claude` extra, which installs
`claude-agent-sdk`:

```bash
pip install "dapr-agents[claude]"
# or
uv add "dapr-agents[claude]"
```

The `claude-agent-sdk` wheels bundle the Claude Code CLI, so they are
about 100 MB. Wheels exist for macOS, glibc Linux (x86_64 and aarch64) and
Windows. There is no musl wheel: on Alpine-based images, install the
`claude` CLI on `PATH` yourself or set `cli_path` in the executor config.
Node.js is not required.

## Setup

From this directory:

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync --active
```

Create a `.env` file with your API key (skip this to use your Claude Code
login instead):

```env
ANTHROPIC_API_KEY=your-api-key
# Optional: pick another model. Defaults to claude-haiku-4-5.
# CLAUDE_MODEL=claude-sonnet-4-5
```

## Run

With the multi-app run file:

```bash
dapr run -f .
```

Or with a single `dapr run`:

```bash
dapr run \
  --app-id claude-executor-app \
  --resources-path ./resources \
  --dapr-http-port 3500 \
  --dapr-grpc-port 50001 \
  -- \
  python app.py
```

`app.py` starts three workflow runs that share one Claude session. By
default it approves the transfer itself. Expected output (trimmed; Claude's
wording varies):

```text
Claude session_id: 6f1c2a0e-...

=== Turn 1: tool call ===
{'role': 'assistant', 'content': 'It is sunny and 22°C in Paris right now.', ...}

=== Turn 2: resumed session ===
{'role': 'assistant', 'content': 'You asked about the weather in Paris.', ...}

=== Approval required ===
  tool        : transfer_money
  arguments   : {'to': 'Alice', 'amount': 25}
  instance_id : 3b9d...
  request_id  : 8e41...
  decision    : approve

=== Turn 3: after approval ===
{"role": "assistant", "content": "Done. I sent $25.00 to Alice for dinner.", ...}
```

Set `APPROVAL_MODE=deny` to reject the transfer instead. Claude receives the
rejection, with the approver's reason ("rejected in app.py"), as the tool
result and explains that the transfer did not happen.

## Send the approval yourself

Run with `APPROVAL_MODE=manual`:

```bash
APPROVAL_MODE=manual dapr run -f .
```

When Claude asks to call `transfer_money`, the workflow publishes an
approval request and waits. `app.py` prints the workflow `instance_id` and
the `request_id`. The workflow waits for a workflow event named
`approval_response_<request_id>`. It auto-denies after 300 seconds (the
`timeout_seconds` on `RequireApproval`).

Send the decision from another terminal, either with the helper script:

```bash
python approval_sender.py <instance_id> <request_id> approve   # or deny
```

or straight to the Dapr sidecar over HTTP:

```bash
curl -X POST \
  "http://localhost:3500/v1.0-beta1/workflows/dapr/<instance_id>/raiseEvent/approval_response_<request_id>" \
  -H "Content-Type: application/json" \
  -d '{"approval_request_id": "<request_id>", "approved": true, "reason": "looks good"}'
```

The pause is a durable Dapr workflow wait, not an open process: the Claude
CLI exits when the run pauses, and the decision resumes the same Claude
session from the state store, on whichever host picks up the workflow.

## Resume a session later

Every run prints its Claude `session_id`. Pass it back to continue that
conversation in a new process:

```bash
SESSION_ID=<session_id-from-a-previous-run> dapr run -f .
```

## How it works

The agent is defined in [`agent.py`](./agent.py). The tools are ordinary
dapr-agents tools, and the approval gate is an ordinary
`before_tool_call` hook:

```python
@tool
def transfer_money(to: str, amount: float) -> str:
    """Transfer an amount of US dollars from the user's account to a recipient."""
    return f"Transferred ${amount:.2f} to {to}."


def require_approval_for_transfers(ctx: ToolHookContext) -> HookDecision:
    if ctx.step_name == "transfer_money":
        return RequireApproval(timeout_seconds=300)
    return Proceed()
```

The executor config only holds Claude-specific settings. Everything else is
configured on the `DurableAgent` as usual:

```python
executor = ClaudeAgentExecutor(
    ClaudeAgentExecutorConfig(
        model="claude-haiku-4-5",
        max_budget_usd=0.50,
    )
)

agent = DurableAgent(
    name="ClaudeAssistant",
    role="Personal Assistant",
    goal="Answer questions and move money only with human approval.",
    instructions=list(INSTRUCTIONS),
    executor=executor,
    tools=[get_weather, transfer_money],
    hooks=Hooks(before_tool_call=[require_approval_for_transfers]),
    state=AgentStateConfig(store=StateStoreService(store_name="agentstatestore")),
    execution=AgentExecutionConfig(max_iterations=6),
)
```

Before every run, `DurableAgent` binds the executor to the agent:

* the rendered system prompt (role, goal and instructions),
* `max_iterations` as Claude's `max_turns`,
* the agent's tools (served to Claude as `mcp__dapr__<tool>`),
* `hooks.before_tool_call`, evaluated in Claude's `PreToolUse` hook,
* a `DaprSessionStore` on the agent's state store, scoped by agent name,
  so Claude transcripts are durable and resume on any host.

Anything you set explicitly in `ClaudeAgentExecutorConfig` (for example
`system_prompt`, `max_turns` or `session_store`) wins over the agent's value.

A few settings matter:

* **No built-in Claude Code tools.** By default the executor turns off
  Claude Code's own tools (`Bash`, `Read`, `Edit`, ...) and does not load
  settings files from the host. Claude can only use the tools you give it.
  Use `builtin_tools` and `allowed_tools` in the config to opt in.
* **One tool call at a time for gated tools.** Claude only pauses a tool
  call cleanly when it is the only call in its message, so the agent's
  instructions ask for one tool at a time. If Claude still batches calls, the
  workflow can go through more than one approval round. Decisions are
  matched to the exact tool call id, so a decision never applies to a
  different call.
* **Budget limits.** `max_iterations` (Claude's `max_turns`) and
  `max_budget_usd` stop a Claude run
  that goes on too long. A stopped run is reported as an error.

## Inspecting Dapr state

Claude transcripts are stored in the `agentstatestore` component under keys
that start with `claude-session:`. Long transcripts are split into chunks.

```bash
redis-cli --scan --pattern 'claude-session:*' | head
```

The agent's own workflow state is stored under keys that start with the
agent name, as in the echo example.

## Running in Kubernetes

* Pass `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`) to the pod from a
  secret. You can also pass it through `env` in `ClaudeAgentExecutorConfig`.
* Use a glibc-based image (for example `python:3.12-slim`), or install the
  `claude` CLI yourself on Alpine.
* The session store needs a Dapr state store that supports ETags and
  first-write concurrency, such as Redis or PostgreSQL.

## Teardown

```bash
dapr stop -f .
# or, if you used a single dapr run:
dapr stop --app-id claude-executor-app
```
