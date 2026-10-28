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

# Durable Deep Agent: A Research Task You Can Crash and Resume

This example builds a deep-agent style research task as a plain `DurableAgent`, with no extra framework:

- A **lead agent** writes a plan first, then works through it one step at a time.
- Each research question goes to a **researcher subagent**, which runs as a child workflow.
- The lead saves its findings as **notes in a Dapr state store** through a tool, so the conversation does not have to carry them.
- The whole run is one Dapr workflow with a fixed instance id. Stop the process halfway, start it again with the same command, and the run continues from the next step.

## Prerequisites

- Python 3.11 to 3.14 and [`uv`](https://docs.astral.sh/uv/)
- The [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/)
- An API key for an LLM provider. The example uses OpenAI by default (`OPENAI_API_KEY`); see [Switching LLM providers](#switching-llm-providers).

All components use SQLite, so nothing else needs to run. No Docker is needed if you use slim mode.

## Setup

From this directory, create a virtual environment and install the dependencies. `dapr run` starts the app with whatever `python3` is on your `PATH`, so keep the environment activated in the shell where you run it:

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync --active
```

Initialize Dapr. Either mode works with the components in `resources/`:

```bash
# With Docker (starts placement, scheduler and Redis for you)
dapr init

# Without Docker
dapr init --slim
```

With `--slim`, start the placement and scheduler services yourself in two separate terminals, and leave them running. Both listen on health port 8080 and metrics port 9090 by default, so the scheduler is moved to 8081 and 9091 here; without that, whichever starts second fails with "address already in use". Start them from this directory. The scheduler is given an explicit data directory (`.dapr-data/`, git-ignored) instead of leaving a `data/` folder wherever it was started:

```bash
~/.dapr/bin/placement
~/.dapr/bin/scheduler --etcd-data-dir ./.dapr-data --healthz-port 8081 --metrics-port 9091
```

Set your key and run the example:

```bash
export OPENAI_API_KEY=<your key>
dapr run -f .
```

You can pass a different topic with `command: ["python3", "deep_research_agent.py", "your topic"]` in `dapr.yaml`.

The run takes about a minute. After each saved note the tool pauses for 8 seconds (`STEP_PAUSE_SECONDS`) so there is time to stop the process in the crash test below.

When the final brief is printed the app exits. `dapr run` keeps its sidecar up until you press `Ctrl+C`.

## Crash test

1. Start the run:

   ```bash
   dapr run -f .
   ```

2. Watch the output until you see the plan saved and at least two `TOOL save_note` lines, for example `TOOL save_note: step 2 saved to research-notes`.

3. Press `Ctrl+C` during the 8 second pause that follows the second `save_note`. The app and its sidecar stop while the run is still unfinished.

4. Start it again with the same command:

   ```bash
   dapr run -f .
   ```

The sample output below assumes the `Ctrl+C` lands in that pause. If it lands earlier or later, the second run simply continues from wherever the run stood: you will see a different retried tool line (or none, if the crash fell between two steps) and different message counts, but still no repeated plan and no repeated finished steps. If you press it after the last note, the second run just prints the final brief.

### What the first run prints

The samples are abridged: timestamps, the app prefix and framework logs are removed. They were produced with a scripted, OpenAI-compatible model, so the lead agent's turns are exactly as shown. With a real model your counts and wording will differ; the final `Final brief:` text is only a placeholder here.

```text
INFO [deep-research] START run deep-research-demo on topic: the main trade-offs of event sourcing
INFO [deep-research] LLM CALL #1 agent=lead messages_in_context=2
INFO [deep-research] TOOL write_plan: saved 3 steps to research-notes
INFO [deep-research] LLM CALL #2 agent=lead messages_in_context=4
INFO [deep-research] LLM CALL #1 agent=researcher messages_in_context=2
INFO [deep-research] LLM CALL #3 agent=lead messages_in_context=6
INFO [deep-research] TOOL save_note: step 1 saved to research-notes
INFO [deep-research] LLM CALL #4 agent=lead messages_in_context=8
INFO [deep-research] LLM CALL #2 agent=researcher messages_in_context=2
INFO [deep-research] LLM CALL #5 agent=lead messages_in_context=10
INFO [deep-research] TOOL save_note: step 2 saved to research-notes
^C
```

### What the second run prints

```text
INFO [deep-research] RESUME run deep-research-demo found in workflow state (status=RUNNING); not rescheduling
INFO [deep-research] TOOL save_note: step 2 saved to research-notes
INFO [deep-research] LLM CALL #1 agent=lead messages_in_context=12
INFO [deep-research] LLM CALL #1 agent=researcher messages_in_context=2
INFO [deep-research] LLM CALL #2 agent=lead messages_in_context=14
INFO [deep-research] TOOL save_note: step 3 saved to research-notes
INFO [deep-research] LLM CALL #3 agent=lead messages_in_context=16
INFO [deep-research] TOOL list_notes: 3 plan steps
...
Final brief:
...
```

### Which lines to look for

- **`RESUME run ... status=RUNNING; not rescheduling`**: the app found the unfinished run in the workflow state and did not start a new one. A first run prints `START run ...` instead.
- **No `TOOL write_plan` line in the second run.** The plan, step 1 and step 2 are in the workflow history. They are read back, not executed again.
- **No repeated `LLM CALL` for the finished steps.** The counters restart at `#1` in each process, so compare `messages_in_context`: the first call after the restart sees 12 messages, which is exactly where the first run stopped (the last call before the crash saw 10, plus the step 2 note exchange). If the finished steps were redone, you would see calls with 2, 4, 6 and so on again. The researcher is also called once, for step 3 only.
- **`TOOL save_note: step 2` appears once more.** The crash hit while step 2's tool was still running (it was in its 8 second pause), so that one activity is retried. Workflow activities run at least once, which is why every tool here is safe to repeat: `save_note` overwrites the same key.

The log lines come from a `before_llm_call` hook, which runs only when a model is actually called, and from the tools themselves. Replayed history never reaches either. With a real model, look for the same things rather than the same numbers: one `RESUME` line, no `write_plan`, a first `messages_in_context` after the restart that is higher than every value logged before the crash, and the researcher called only for the steps that were not finished.

One more thing you will see in the model provider's logs: each agent run ends with one extra summarization call that the framework makes for its memory. It is not a reasoning step and is not counted by the hook.

## How it works

```text
lead agent (workflow: dapr.agents.lead.workflow)
 ├─ write_plan      -> plan saved to the research-notes store
 ├─ researcher      -> child workflow of the researcher agent, one question at a time
 ├─ save_note       -> note saved to the research-notes store
 ├─ list_notes / read_note -> read the notes back to write the final brief
```

- `DurableAgent` runs each LLM call and each tool call as a workflow activity. The workflow records every result in its history.
- Passing `researcher` in the lead's `tools` list turns it into a tool. A call runs the researcher's own workflow as a child workflow. Both agents share one `WorkflowRuntime` so they live in one process.
- `AgentRunner.run(..., instance_id="deep-research-demo", wait=False)` starts the run under a fixed id. On the next start, the app checks the workflow state for that id and, if it exists, only starts the worker. The Dapr workflow engine then continues the run by itself.
- The components are found automatically: `agent-workflow` (workflow history, also the actor state store) and `agent-memory` (conversation memory) by name, and the model by its `conversation.*` type. The notes store `research-notes` is used by the tools.

## Where the notes live and why

The notes live in the `research-notes` state store, not in the conversation. Three things follow from that:

- **They survive the restart.** The store is a SQLite file, `research-notes.db`, next to the app. Stopping the process does not touch it.
- **The plan position comes from the workflow.** The agent never keeps a "current step" in memory or in a file. Where the run stands is the workflow history: which LLM calls and tool calls already finished. After a restart the engine replays that history and runs the next unfinished action.
- **The two cannot drift apart.** A note is written by a tool call inside the workflow, so a step is either recorded in the history (and its note is in the store) or it is not recorded and gets run again. The one gap is a crash between writing the note and recording the result. Then the tool runs again and overwrites the same key with the same kind of content, so the store ends up consistent with the history.

Because notes are in the store, the conversation stays short. The lead only reads them back at the end with `list_notes` and `read_note`.

You can look at the notes yourself while the run is stopped. The keys are prefixed with the app id (`deep-research-app||`) and then the run id, and the values are stored as JSON text:

```bash
sqlite3 research-notes.db "select key, value from state"
```

## Starting over

A finished or stopped run keeps its id. If you start the app again with the id of a run that already completed, failed or was terminated, it prints that status (and the brief or the error) and exits without starting anything.

The simplest way to start a new run is a new id:

```bash
RESEARCH_RUN_ID=my-second-run dapr run -f .
```

If the previous run is **not finished** (for example you stopped it in the crash test and would rather abandon it than resume it), the app refuses to start a different id and tells you which run is still open. Starting the worker would resume the old run too, and its tools would write into the new run's note keys. Either resume the old run with the same command, or end it first:

1. Start the app with the old id: `RESEARCH_RUN_ID=<old id> dapr run -f .`
2. While it is running, end the run from a second terminal: `dapr workflow terminate -a deep-research-app <old id>`. This works only while the app that hosts the run is up. The app prints that the run ended with status TERMINATED and exits.
3. Start the new run with a new id.

A tool call that was in flight when you terminated the old run can still be redelivered once when the next process starts, and it writes one stray note under the new run's id. The new run overwrites it when it reaches that step.

To wipe everything instead, stop the app and delete the `*.db`, `*.db-shm` and `*.db-wal` files in this folder. The scheduler keeps the jobs of old runs outside those files, and a leftover job could wake a run you deleted. In slim mode, stop the scheduler and delete `.dapr-data/` as well. With `dapr init` (Docker), the scheduler's data lives in the scheduler container, not in this folder, so deleting files here does not clear it. A new `RESEARCH_RUN_ID` avoids all of this and is usually the better choice.

## Switching LLM providers

The model is chosen only by the Dapr conversation component in `resources/llm-provider.yaml`. The agent uses the first conversation component it finds in `resources/`, so keep exactly one conversation component there (the name does not matter); the code never mentions a provider.

Anthropic:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: llm-provider
spec:
  type: conversation.anthropic
  version: v1
  metadata:
  - name: key
    envRef: ANTHROPIC_API_KEY
  - name: model
    value: claude-sonnet-4-5
```

Ollama, running locally:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: llm-provider
spec:
  type: conversation.ollama
  version: v1
  metadata:
  - name: model
    value: llama3.1
```

Use a model that supports tool calling. See the [Dapr conversation components](https://docs.dapr.io/reference/components-reference/supported-conversation/) for all providers.

## Files

| File | Purpose |
|---|---|
| `deep_research_agent.py` | The lead and researcher agents, the note tools, and the resume logic |
| `dapr.yaml` | Multi-app run file used by `dapr run -f .` |
| `resources/llm-provider.yaml` | LLM provider (the only place to change it) |
| `resources/agent-workflow.yaml` | Workflow history and actor state (SQLite) |
| `resources/agent-memory.yaml` | Conversation memory (SQLite) |
| `resources/research-notes.yaml` | Research notes (SQLite) |
