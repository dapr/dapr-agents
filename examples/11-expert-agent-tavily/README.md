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

# Expert agent — Chainlit UI with RAG-via-hook (Tavily)

This example demonstrates how to attach a `before_llm_call` hook to a `DurableAgent` to inject fresh web context into every LLM turn — automatically, without the model needing to choose to call a `web_search` tool.

## Why a hook, not a tool?

A `web_search` tool depends on the LLM *deciding* to invoke it. The model often won't, especially for questions it thinks it knows ("What's the latest Dapr release?" → confident, wrong answer from training data).

A `before_llm_call` hook fires for **every** LLM call. The user's question gets enriched whether the model thinks it needs help or not. The hook also lives with the agent, so the same enrichment works behind any UI (Chainlit here, FastAPI, pub/sub, etc.).

## How it works

1. The user types a question in Chainlit.
2. `runner.run(agent, {"task": ...})` schedules an agent workflow.
3. Inside the `call_llm` activity, the `before_llm_call` hook fires:
   - The hook reads the last user message from the LLM call's `messages` payload.
   - It calls `tavily.search(...)` for fresh web context.
   - It returns `Mutate(payload=...)` to splice the results into the prompt as a system message just before the user's question.
4. The (now-enriched) prompt is sent to the LLM, which answers using the injected context.
5. The activity's recorded output is what the workflow durably remembers — Tavily is only called once per turn, even on replay.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) package manager
- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) (run `dapr init` once)
- OpenAI API key
- [Tavily API key](https://tavily.com) (free tier supports 1000 searches/month)

## Setup

```bash
uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv sync --active
dapr init                     # only needed once per machine
```

Create a `.env` file in this directory:

```env
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## Run

```bash
uv run dapr run --app-id expert-agent --resources-path ./resources -- chainlit run app.py -w
```

Open <http://localhost:8000>.

Try questions the LLM's training cutoff would normally block:

- *"What's the latest Dapr release version and what's new in it?"*
- *"Who won the last F1 race?"*
- *"Summarize this week's biggest open-source security advisories."*

Watch the agent log — you'll see `[hook] Tavily search: <query>` every turn, confirming the hook is firing.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Chainlit entrypoint; lazy-inits the agent and runner |
| `agent.py` | `DurableAgent` factory; registers the Tavily hook |
| `hooks.py` | The `before_llm_call` hook that calls Tavily and rewrites the prompt |
| `resources/conversationmemory.yaml` | Conversation history state store |
| `resources/workflowstate.yaml` | Workflow state store (must have `actorStateStore: "true"`) |
| `resources/registrystate.yaml` | Agent registry state store |

## What to try next

- Tweak `enrich_with_tavily` in `hooks.py` — cache results per-query, filter by domain, change `max_results`.
- Add a second hook (e.g. `after_llm_call` that logs the final answer) by appending it to the `Hooks(...)` arglist in `agent.py`.
- Swap `OpenAIChatClient` for any other Dapr-supported provider in `agent.py` — the hook keeps working unchanged.
