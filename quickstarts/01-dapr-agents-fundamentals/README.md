# Dapr Agents Fundamentals

This quickstart introduces the core concepts of Dapr Agents and walks you through progressively more advanced examples. You'll learn how to build agents that use memory and tools, how to run durable agents backed by workflows, and how to orchestrate multiple agents in deterministic workflows.

You will learn how to:
 
1. **[Use native LLM client](#1-llm-client)**
2. **[Build a simple agent with LLM](#2-agent-with-llm)**
3. **[Build a simple agent with LLM + tools](#3-agent-with-llm-and-tools)**
4. **[Use MCP tools loaded over STDIO](#4-agent-with-mcp-tools)**
5. **[Add memory with persistence to the agent](#5-agent-with-memory-and-tools)**
6. **[Run an agent as a durable workflow](#6-durable-agent-serve)**
7. **[Trigger durable agents using pub/sub messages](#7-durable-agent-subscribe)**
8. **[Use deterministic workflows that call LLMs](#8-workflow-with-llm-activities)**
9. **[Orchestrate multiple agents inside a workflow](#9-workflow-with-agent-activities)**
10. **[Enable distributed tracing for agents with Zipkin](#10-durable-agent-trace-zipkin)**
 
These examples form the foundation of the Dapr Agents programming model and illustrate how LLM reasoning, tool execution, durable workflows, and agent coordination fit together.

---
## Prerequisites

- Python 3.10+ (https://www.python.org/downloads/)
- Docker (https://docs.docker.com/get-docker/)
- Dapr CLI (https://docs.dapr.io/getting-started/install-dapr-cli/)
- uv package manager (https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI API key (https://platform.openai.com/api-keys) or another LLM provider

## Environment Setup

<details open>
<summary><strong>Option 1: Using uv (Recommended)</strong></summary>

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
 
</details>

<details>
<summary><strong>Option 2: Using pip</strong></summary>

```bash
python3.10 -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

</details>

## OpenAI Configuration

> Warning
> These examples require an OpenAI API key.

The quickstart uses an OpenAI conversation component located in the `resources` directory. You can replace the provider with Anthropic, Ollama, or others. [See here how to configure another component](https://docs.dapr.io/reference/components-reference/supported-conversation/)

### Component Configuration

Update `resources/llm-provider.yaml`:

```yaml
metadata:
  - name: key
    value: "{{OPENAI_API_KEY}}"
```

Replace `OPENAI_API_KEY` with your actual OpenAI API key.

---

# 1. LLM Client

This example shows the simplest way to call an LLM using the Dapr Chat Client, which sends prompts through the Dapr Conversation API. It’s a minimal starting point before introducing agents in later examples.

```bash
dapr run --app-id llm-client --resources-path resources -- python 01_llm_client.py
```

## Expected Behavior
Running the script sends the prompt to the LLM provider and prints the model’s reply. By default, the Conversation API component uses OpenAI, but you can switch providers by updating the component YAML.

## How This Works

1. The DaprChatClient sends the prompt to the Dapr sidecar using the Conversation API under the hood.
2. The Dapr sidecar uses the configured OpenAI component file to forward the prompt to the LLM provider and returns the generated response to your application.

## How to Extend This Example

Dapr Agents also include native LLM clients for other modalities (e.g., audio), which you can explore when your application requires more than simple chat.

---

# 2. Agent with LLM

This example introduces the basic concept of a Dapr Agent. An agent wraps an LLM with a name, role, and instructions that define how it should behave. Unlike the previous example—where you called the LLM directly—an agent provides a reusable interface you can trigger multiple times, and it will consistently act according to its assigned role.

```bash
dapr run --app-id agent-llm --resources-path resources -- python 02_agent_llm.py
```

## Expected Behavior
Running the script constructs an agent with a defined role and behavior, sends it a weather-related prompt, and prints its response. Because the agent has no tools, the LLM simply makes a best-effort guess about the weather based on its internal knowledge.

## How This Works

1. The agent is created with a name, a role, and a set of instructions that act as system-level guidance for how it should respond.
2. Internally, the agent uses the DaprChatClient, so each agent invocation sends a prompt through the Dapr Conversation API and receives the LLM’s response.

## How to Extend This Example

Modify the agent’s role or instructions and observe how its behavior changes when answering prompts.

---

# 3. Agent with LLM and Tools

This example shows how to quickly create an agent with a custom prompt, backed by the Dapr Conversation API, and how to expose a local Python function as a tool the agent can call during reasoning. It demonstrates the simplest way to run an agent locally as a regular Python program while benefiting from Dapr’s LLM abstraction.

```bash
dapr run --app-id agent-llm --resources-path resources -- python 03_agent_llm_tools.py
```

## Expected Behavior
When you run the script, the agent receives a weather question, invokes a local tool to retrieve the temperature, and uses the LLM to produce a natural-language answer.
 
## How This Works

1. The agent sends prompts to the Dapr Conversation API, which routes them to the configured LLM provider without requiring changes to your application code.
2. A Python function is registered as a tool, and the agent executes it when the LLM decides a tool call is needed.
3. The interaction runs as a single-turn exchange with no persistence, serving as the minimal foundation for later examples.

## How to Extend This Example

* In addition to using the Conversation API, Dapr Agents also provide **native LLM clients** when you need other modality and adcanced features beyond simple chat. To explore this, see these examples [LLM Call quickstart](../02-llm-call-open-ai/README.md).
* The functions you expose as tools can call out to **remote services using the Dapr client**, gaining resiliency (retries, timeouts, circuit breakers) and built-in observability through Dapr’s features.
 
---

# 4. Agent with MCP Tools  

This example is very similar to the previous one, except that the agent does not use hard-coded Python functions as tools. Instead, it dynamically discovers its tools from an MCP (Model Context Protocol) server running locally over STDIO, allowing tools to be added or modified without changing the agent code.

```bash
dapr run --app-id agent-mcp --resources-path resources -- python 04_agent_mcp_tools.py
```

## Expected Behavior
When you run the script, the agent queries the MCP server for available tools, invokes the MCP-provided weather tool to answer the question, and uses the LLM to produce the final response.
 
## How This Works

1. The agent connects to an MCP server over STDIO, allowing tools to be negotiated and loaded dynamically at runtime.
2. The weather tools are served by the local MCP script (`mcp_tools.py`), and the agent invokes them when the LLM requests a tool call.
3. The LLM call still goes through the Dapr Conversation API, giving the same provider abstraction as in Example 3 but with a more flexible tool architecture.

## How to Extend This Example

* To see how to connect to **remote MCP servers** (via SSE or Streamable HTTP) instead of STDIO, check out:

  * [MCP Client with SSE](../07-agent-mcp-client-sse)
  * [MCP Client with Streamable HTTP](../07-agent-mcp-client-streamablehttp)

---

# 5. Agent with Memory

This example shows how to create an agent that can store and recall its full conversation history across multiple interactions using a Dapr state store. By persisting the session history, the agent can continue a multi-turn dialog and provide answers informed by prior messages.

```bash
dapr run --app-id agent-memory --resources-path resources -- python 05_agent_memory.py
```

## Expected Behavior
The script runs two prompts in sequence: the agent answers the initial weather question and persists the entire conversation history to the external state store. When the second prompt is sent—whether during the same run or after restarting the agent—it loads the stored session history and responds using that previously saved context.

## How This Works

1. The agent persists the full conversation history to a Dapr state store after each interaction, making the session durable across process restarts.
2. Each call to `weather_agent.run()` retrieves any previously stored history, allowing the agent to continue the conversation seamlessly.
3. The agent still performs tool calls as in earlier examples, but the LLM’s response now considers the restored session history.

## How to Extend This Example

* You can switch to any supported Dapr state store (Redis, Azure Cosmos DB, PostgreSQL, etc.) by updating the component YAML without modifying the code. See:

  * **State Management Overview:** [https://docs.dapr.io/developing-applications/building-blocks/state-management/state-management-overview/](https://docs.dapr.io/developing-applications/building-blocks/state-management/state-management-overview/)
  * **Supported State Stores:** [https://docs.dapr.io/reference/components-reference/supported-state-stores/](https://docs.dapr.io/reference/components-reference/supported-state-stores/)
 
---

# 6. Durable Agent Serve

This example turns the previous agent into a durable agent backed by the Dapr Workflow engine. Instead of running interactions in-process, every step of the agent’s execution is persisted to durable storage, allowing long-running interactions to survive interruptions. The agent exposes an HTTP endpoint to start a new workflow and provides a way to query progress or retrieve the final result at any time.
```bash
dapr run --app-id durable-agent --resources-path resources -- python 06_durable_agent_http.py
```

On a different terminal, trigger the agent:

```bash
curl -i -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in London?"}'
```
You will receive a WORKFLOW_ID in response. Query the result:

```bash
curl -i -X GET http://localhost:8001/run/WORKFLOW_ID
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

# 7. Durable Agent Subscribe

This example takes the same durable agent behavior from the previous example, but instead of exposing an HTTP endpoint, it uses pub/sub. With this setup, the durable agent runs in the background as an ambient agent and listens for incoming events on a message topic. When a message arrives, it automatically starts a workflow execution. 

The agent code remains unchanged; only the AgentRunner configuration switches from REST to pub/sub.
```bash
dapr run --app-id durable-agent-subscriber --resources-path resources --dapr-http-port 3500 -- python 07_durable_agent_pubsub.py
```

On a different terminal, publish a message to the subscribed topic:

```bash
dapr publish --publish-app-id durable-agent-subscriber --pubsub message-pubsub --topic weather.requests --data '{"task": "What is the weather in London?"}'
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

# 8. Workflow with LLM Activities
This example does not use an agent. Instead, it demonstrates how to create a Dapr workflow that performs LLM calls in a deterministic, durable sequence.

```bash
dapr run --app-id workflow-llms --resources-path resources -- python 08_workflow_llm.py
```

## Expected Behavior

The workflow generates a short outline for the given topic using an LLM, then uses that outline to produce a short blog post. Both steps run as durable activities, so the workflow can restart without repeating completed LLM calls.

## How This Works

1. The workflow first performs an LLM-backed activity that generates an outline from the topic. This activity is decorated with `@llm_activity`, a Dapr Agents annotation that simplifies workflow activities by automatically wiring in the LLM client and performing the model invocation for you.
2. The resulting outline is passed to a second `@llm_activity`-decorated activity, which uses the LLM to generate the final blog post. This output is returned as the result of the workflow.

## How to Extend This Example
* Modify the workflow to include additional activities that do not interact with LLMs, such as inserting validation steps, transformations, or business logic between LLM activities.
* Use structured output to enforce schema-based responses from the LLM for predictable and validated workflow inputs. To see structured output and validation, refer to the [LLM Call quickstart](../02-llm-call-open-ai/README.md).

---

# 9. Workflow with Agent Activities

This example shows how a workflow can invoke entire agents as workflow activities, allowing you to orchestrate multi-step agent reasoning in a durable and deterministic way. Unlike previous examples where activities called LLMs directly, this workflow delegates each step to an agent with tools and memory, while the workflow engine provides durability and reliable progression.

```bash
dapr run --app-id workflow-agents --resources-path resources -- python 09_workflow_agents.py
```

## Expected Behavior

When the workflow runs, it first delegates the request to a triage agent, which gathers customer information using tools and produces a summary. It then passes that summary to an expert agent, which generates a final recommendation. Both steps run under a durable workflow, so if the process is interrupted, it resumes from the last completed activity even though the agents themselves are not durable.

## How This Works

1. The workflow invokes each agent using activities decorated with @agent_activity, which handles calling the agent and returning structured output.
2. The triage activity runs first, producing a summary based on customer data and the issue description.
3. The output of the triage agent is passed into the expert agent activity to generate the final recommendation.
4. Although agents can use tools and maintain their own memory, the workflow execution is what provides durability: if interrupted, it restarts from the last completed step.

## How to Extend This Example
Add additional workflow activities—some invoking agents, others performing business logic or LLM steps to create richer multi-stage workflows.

---

# 10. Durable Agent Trace (Zipkin)

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
dapr run --app-id durable-agent-trace --resources-path resources -- python 10_durable_agent_tracing.py
```

## Expected Behavior

When the script runs, the durable agent executes its workflow in-process and emits tracing spans for every LLM call, tool call, and workflow step. These spans appear in Zipkin alongside the spans generated by the Dapr sidecar, giving you a unified, end-to-end view of the agent’s reasoning and workflow execution.

## How This Works

1. The Dapr sidecar automatically intercepts the durable workflow execution and emits workflow-level spans for each step, retry, and state transition, giving you visibility into the orchestration layer.
2. The application enables Dapr Agents instrumentation, which intercepts agent-level operations—including LLM invocations, tool calls, memory reads/writes, and decision steps—and records them as additional spans. Once the agent runs, you can open the Zipkin UI at the URL above and inspect the complete trace to see exactly how the agent behaves and how these spans are connected.

## How to Extend This Example
Open the Zipkin UI at the URL above and explore the full trace to see how the workflow spans and agent spans connect end-to-end.

---

# Other Dapr Agent Examples
If you want to coordinate multiple agents that run in separate applications or communicate through Pub/Sub, check out the [multi-agent workflows quickstart](../05-multi-agent-workflows/README.md).

---

# Troubleshooting

1. **API Key Issues**: If you see an authentication error, verify your LLM provider key is in the `llm-provider.yaml` file
2. **Python Version**: If you encounter compatibility issues, make sure you're using Python 3.10+
3. **Environment Activation**: Ensure your virtual environment is activated before running examples
4. **Import Errors**: If you see module not found errors, verify that `pip install -r requirements.txt` completed successfully

# Next Steps

Learn how to use structured outputs with LLMs in the [LLM Call quickstart](../02-llm-call-open-ai/README.md).
 