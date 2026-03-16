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

# Dapr Agents Fundamentals

This quickstart introduces the core concepts of Dapr Agents and walks you through progressively more advanced examples. You'll learn how to run durable agents backed by workflows, and how to orchestrate multiple agents in deterministic workflows.

You will learn how to:

1. **[Use native LLM client](#1-llm-client)**
2. **[Run an agent as a durable workflow](#2-durable-agent-serve)**
3. **[Trigger durable agents using pub/sub messages](#3-durable-agent-subscribe)**
4. **[Use deterministic workflows that call LLMs](#4-workflow-with-llm-activities)**
5. **[Orchestrate multiple agents inside a workflow](#5-workflow-with-agent-activities)**
6. **[Enable distributed tracing for agents with Zipkin](#6-durable-agent-trace-zipkin)**
7. **[Hot-reload agent configuration at runtime](#7-durable-agent-hot-reload)**

These examples form the foundation of the Dapr Agents programming model and illustrate how LLM reasoning, tool execution, durable workflows, and agent coordination fit together.

---
## Prerequisites

- Python >= 3.11 (https://www.python.org/downloads/)
- Docker (https://docs.docker.com/get-docker/)
- Dapr CLI (https://docs.dapr.io/getting-started/install-dapr-cli/)
- uv package manager (https://docs.astral.sh/uv/getting-started/installation/)
- Ollama (https://ollama.com/) **or** an OpenAI API key (https://platform.openai.com/api-keys) or another LLM provider

## Environment Setup

<details open>
<summary><strong>Install dependencies</strong></summary>

```bash
uv venv
# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
uv sync --active
```

Ensure Dapr is running locally

```bash
dapr init
```

</details>

## LLM Configuration

By default, the quickstart uses [Ollama](https://ollama.com/) so you can run everything locally without an API key.

### Default: Ollama (Local)

1. **Install and start Ollama:**

   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull a model with tool-calling support:**

   ```bash
   ollama serve    # Start the server (skip if already running)
   ollama pull qwen3:0.6b
   ```

3. **Set environment variables before running any quickstart:**

   ```bash
   export OLLAMA_ENDPOINT=http://localhost:11434/v1
   export OLLAMA_MODEL=qwen3:0.6b
   ```

   The `resources/llm-provider.yaml` component resolves `{{OLLAMA_ENDPOINT}}` and `{{OLLAMA_MODEL}}` from your environment automatically.

> **Tip:** For more reliable tool calling, use a larger model such as `qwen2.5:3b` or `llama3.1:8b`.

### Alternative: OpenAI

To use OpenAI instead, replace `resources/llm-provider.yaml` with:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: llm-provider
spec:
  type: conversation.openai
  version: v1
  metadata:
  - name: key
    value: "{{OPENAI_API_KEY}}"
  - name: model
    value: "gpt-4o-mini"
```

### Alternative: Other Providers

Dapr supports Anthropic, Mistral, and other LLM providers through the Conversation API. Replace the component type and metadata while keeping the component name as `llm-provider`. See the [Dapr Conversation component reference](https://docs.dapr.io/reference/components-reference/supported-conversation/) for the full list of supported providers and their configuration.

---

# 1. LLM Client

This example shows the simplest way to call an LLM using the Dapr Chat Client, which sends prompts through the Dapr Conversation API. It’s a minimal starting point before introducing agents in later examples.

```bash
uv run dapr run --app-id llm-client --resources-path resources -- python 01_llm_client.py
```

## Expected Behavior
Running the script sends the prompt to the LLM provider and prints the model’s reply. By default, the Conversation API component uses Ollama, but you can switch to OpenAI or other providers by updating the component YAML (see [LLM Configuration](#llm-configuration)).

## How This Works

1. The DaprChatClient sends the prompt to the Dapr sidecar using the Conversation API under the hood.
2. The Dapr sidecar uses the configured conversation component to forward the prompt to the LLM provider (Ollama by default) and returns the generated response to your application.

## How to Extend This Example

Dapr Agents also include native LLM clients for other modalities (e.g., audio), which you can explore when your application requires more than simple chat.

---

# 2. Durable Agent Serve

This example introduces the `DurableAgent`, a workflow-native agent backed by the Dapr Workflow engine. Every step of the agent’s execution is persisted to durable storage, allowing long-running interactions to survive interruptions. The agent exposes an HTTP endpoint to start a new workflow and provides a way to query progress or retrieve the final result at any time.
```bash
uv run dapr run --app-id durable-agent --resources-path resources -- python 02_durable_agent_http.py
```

On a different terminal, trigger the agent:

```bash
curl -i -X POST http://localhost:8001/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in London?"}'
```
You will receive a WORKFLOW_ID in response. Query the result:

```bash
curl -i -X GET http://localhost:8001/agent/instances/WORKFLOW_ID
```

Replace `WORKFLOW_ID` with the ID returned from the POST request.

## Expected Behavior

The agent exposes a REST endpoint, accepts a prompt, and returns a workflow ID that represents a durable execution. You can query this workflow at any time—even after stopping and restarting the agent—and it will resume exactly where it left off. The agent performs an LLM call and a tool call as part of completing the workflow and produces a final result.

## How This Works

1. The agent schedules the prompt as a workflow execution and persists every step to durable storage.
2. The agent creates a workflow activity to perform the LLM interaction and determine whether a tool call is needed.
3. The agent creates another workflow activity to perform the tool call.
4. The agent creates another workflow activity to return the tool call result to the LLM and complete the reasoning step.
5. The agent finishes the execution, persisting every interaction and the final result. The workflow engine ensures reliable progression so no LLM or tool call is repeated unless required.


**Testing durability:**

This example includes a different tool, **SlowWeatherTool**, which intentionally waits five seconds before returning a result. This delay allows you to interrupt the agent mid-execution and verify that the workflow engine resumes from the same point after the agent restarts.

To test this:

1. Trigger the agent with a prompt using the POST command shown above.
2. During the 5-second delay inside **SlowWeatherTool**, stop the agent by pressing **Ctrl+C**.
3. Restart the agent using the same `dapr run` command.
4. Query the workflow using the same `WORKFLOW_ID`; you will see that it continues from the step it was on—**without starting over, without repeating the LLM call, and without requiring a new prompt**.
5. Once the workflow finishes, the GET request will show the completed result.

In summary, the workflow engine preserves execution state across restarts, enabling reliable continuation of long-running agent interactions.

## How to Extend This Example

* Add custom workflow activities for business logic or integrations.
* Combine multiple agents inside the same durable workflow.
* To see multi-step workflows with LLM interactions, refer to the [LLM-based workflows quickstart](../04-llm-based-workflows/README.md).

---

# 3. Durable Agent Subscribe

This example takes the same durable agent behavior from the previous example, but instead of exposing an HTTP endpoint, it uses pub/sub. With this setup, the durable agent runs in the background as an ambient agent and listens for incoming events on a message topic. When a message arrives, it automatically starts a workflow execution.

The agent code remains unchanged; only the AgentRunner configuration switches from REST to pub/sub.
```bash
uv run dapr run --app-id durable-agent-subscriber --resources-path resources --dapr-http-port 3500 -- python 03_durable_agent_pubsub.py
```

On a different terminal, publish a message to the subscribed topic:

```bash
dapr publish --publish-app-id durable-agent-subscriber --pubsub agent-pubsub --topic weather.requests --data '{"task": "What is the weather in London?"}'
```

## Expected Behavior

The agent listens to the weather.requests topic and, when a message is published, begins a durable workflow execution using the same logic as in the previous example. You can restart the agent at any time during execution, and the workflow will continue from the exact step where it was interrupted.

## How This Works

1. The agent runs as a durable agent subscribed to a pub/sub topic and listens for incoming events.
2. A message is published to the topic using the dapr publish command.
3. The agent runner receives the event and forwards it to the durable agent.
4. The message triggers a workflow execution, which performs the LLM and tool-call activities with durable state persisted at every step.

## How to Extend This Example

Try publishing multiple messages to the topic and observe the agent process each message as an independent durable workflow execution.

---

# 4. Workflow with LLM Activities
This example does not use an agent. Instead, it demonstrates how to create a Dapr workflow that performs LLM calls in a deterministic, durable sequence.

```bash
uv run dapr run --app-id workflow-llms --resources-path resources -- python 04_workflow_llm.py
```

## Expected Behavior

The workflow generates a short outline for the given topic using an LLM, then uses that outline to produce a short blog post. Both steps run as durable activities, so the workflow can restart without repeating completed LLM calls.

## How This Works

1. The workflow first performs an LLM-backed activity that generates an outline from the topic. This activity uses a direct LLM call, optionally with schema validation, for predictable and validated output.
2. The resulting outline is passed to a second LLM-backed activity, which uses the LLM to generate the final blog post. This output is returned as the result of the workflow.

## How to Extend This Example
* Modify the workflow to include additional activities that do not interact with LLMs, such as inserting validation steps, transformations, or business logic between LLM activities.
* Use structured output to enforce schema-based responses from the LLM for predictable and validated workflow inputs. To see structured output and validation, refer to the [LLM Call quickstart](../02-llm-call-open-ai/README.md).

---

# 5. Workflow with Agent Activities

This example shows how a workflow can invoke entire agents as child workflows, allowing you to orchestrate multi-step agent reasoning in a durable and deterministic way. Unlike previous examples where activities called LLMs directly, this workflow delegates each step to an agent with tools and memory, while the workflow engine provides durability and reliable progression.

```bash
uv run dapr run -f 05_workflow_agents.yaml
```

## Expected Behavior

When the workflow runs, it first delegates the request to a triage agent, which gathers customer information using tools and produces a summary. It then passes that summary to an expert agent, which generates a final recommendation. Both steps run under a durable workflow, so if the process is interrupted, it resumes from the last completed activity even though the agents themselves are not durable.

## How This Works

1. The workflow invokes each agent by calling agent-backed activities as child workflows using `ctx.call_child_workflow`, which handles calling the agent and returning structured output.
2. The triage activity runs first, producing a summary based on customer data and the issue description.
3. The output of the triage agent is passed into the expert agent activity to generate the final recommendation.
4. Although agents can use tools and maintain their own memory, the workflow execution is what provides durability: if interrupted, it restarts from the last completed step.

## How to Extend This Example
Add additional workflow activities—some invoking agents, others performing business logic or LLM steps to create richer multi-stage workflows.

---

# 6. Durable Agent Trace (Zipkin)

This example shows how to enable end-to-end tracing for a durable agent using OpenTelemetry and Zipkin. While the Dapr sidecar automatically emits workflow-related spans for each durable execution, this example extends the tracing model by adding application-level tracing inside the agent itself, allowing you to observe agent-specific steps such as LLM calls, tool calls, and memory operations alongside the workflow spans.

## Run Zipkin Backend

Dapr CLI installs and runs Zipkin by default. You can check whether it is running by visiting:

http://localhost:9411/

If Zipkin is not running, start it manually:

```bash
# Start Zipkin locally
docker run -d -p 9411:9411 openzipkin/zipkin
```

Now run the durable agent with tracing enabled and prompting included:

```
uv run dapr run --app-id durable-agent-trace --resources-path resources -- python 06_durable_agent_tracing.py
```

## Expected Behavior

When the script runs, the durable agent executes its workflow in-process and emits tracing spans for every LLM call, tool call, and workflow step. These spans appear in Zipkin alongside the spans generated by the Dapr sidecar, giving you a unified, end-to-end view of the agent’s reasoning and workflow execution.

## How This Works

1. The Dapr sidecar automatically intercepts the durable workflow execution and emits workflow-level spans for each step, retry, and state transition, giving you visibility into the orchestration layer.
2. The application enables Dapr Agents instrumentation, which intercepts agent-level operations—including LLM invocations, tool calls, memory reads/writes, and decision steps—and records them as additional spans. Once the agent runs, you can open the Zipkin UI at the URL above and inspect the complete trace to see exactly how the agent behaves and how these spans are connected.

## How to Extend This Example
Open the Zipkin UI at the URL above and explore the full trace to see how the workflow spans and agent spans connect end-to-end.

---

# 7. Durable Agent Hot-Reload

This example shows how to subscribe a durable agent to a Dapr Configuration Store so that its persona (role, goal, instructions) and other settings can be updated at runtime without restarting the process. When a value changes in the backing store (e.g. Redis), the agent picks up the update automatically.

First, ensure the `runtime-config` component is available in your resources path. You can use the one provided in `resources/configstore.yaml`. For supported configuration store backends, see the [Dapr docs](https://docs.dapr.io/reference/components-reference/supported-configuration-stores/).

```bash
dapr run --app-id hot-reload-agent --resources-path resources -- python 07_durable_agent_hot_reload.py
```

In a separate terminal, update a configuration value directly in Redis:

```bash
redis-cli SET agent_role "New Hot-Reloaded Role"
```

## Expected Behavior

The agent starts with its initial role (`Original Role`) and subscribes to the Dapr configuration store for the keys `agent_role`, `agent_goal`, and `agent_instructions`. When you update a value in Redis, the agent's profile updates in-place and the change is visible in the periodic log output—without restarting.

## How This Works

1. The agent is initialized with a `RuntimeSubscriptionConfig` that specifies the configuration store name and the keys to watch.
2. When the runner calls `subscribe()` (or `serve()`), the agent loads existing values and subscribes to the Dapr Configuration API using `subscribe_configuration`, which streams updates from the backing store, then starts the workflow runtime.
3. When a configuration key changes, the `_config_handler` in `AgentBase` receives the update and applies it to the agent's profile, LLM settings, or component references.
4. If a registry store is configured, the agent re-registers its updated metadata automatically.

## How to Extend This Example

* Update multiple keys at once by setting a JSON object as the configuration value.
* Add additional keys for LLM settings (`llm_model`, `llm_provider`) to swap models at runtime.
* For the full list of supported configuration keys, see the [hot-reload example README](../examples/09-durable-agent-hot-reload/README.md).

---

# Other Dapr Agent Examples
If you want to coordinate multiple agents that run in separate applications or communicate through Pub/Sub, check out the [multi-agent workflows quickstart](../05-multi-agent-workflows/README.md).

---

# Troubleshooting

1. **Ollama not responding**: Ensure `ollama serve` is running and the model is pulled (`ollama pull qwen3:0.6b`)
2. **Environment variables**: Verify `OLLAMA_ENDPOINT` and `OLLAMA_MODEL` are exported (or `OPENAI_API_KEY` if using OpenAI)
3. **API Key Issues**: If using OpenAI and you see an authentication error, verify your key is set in the `llm-provider.yaml` component
4. **Python Version**: If you encounter compatibility issues, make sure you're using Python 3.10+
5. **Environment Activation**: Ensure your virtual environment is activated before running examples
6. **Import Errors**: If you see module not found errors, verify that `uv sync --active` completed successfully

# Other Dapr Agents Examples
If you want to see more Dapr Agents examples, check out the [examples](../examples/) folder.
