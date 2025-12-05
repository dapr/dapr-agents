# Hello World with Dapr Agents

This quickstart introduces Dapr Agents through simple examples that show how to build agents with memory and tools, run durable agents, and orchestrate agents through workflows. You will learn:

1. How to create agents with memory, tools, and LLMs backed by Dapr
2. How to build durable agents that survive restarts through Dapr workflows
3. How to use deterministic workflows that interact with LLMs
4. How to coordinate multiple agents using deterministic workflows

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

Update `resources/openai.yaml`:

```yaml
metadata:
  - name: key
    value: "{{OPENAI_API_KEY}}"
```

Replace `OPENAI_API_KEY` with your actual OpenAI API key.

# Examples

---

# 1. Agent with Memory and Tools

This example shows how to create an agent that uses memory and interacts with an LLM. The LLM is accessed through the Dapr Conversation API and memory is stored using the Dapr State Store API.

```bash
dapr run --app-id agent-memory --resources-path resources -- python 01_agent_with_memory.py
```

## Expected Behavior
When you run the agent, it answers a weather question by calling its tool and then remembers information from the conversation, such as the user’s name, and uses that memory in a later response.

## How This Works

1. The agent uses the Dapr Conversation API which allows you to switch LLM providers without changing code.
2. The agent stores conversation history in the Dapr state store which lets it resume context after restart.
3. The agent calls a local tool function for weather information.

## How to Extend This Example

* Tools can be dynamically discovered through MCP servers. See the [MCP client quickstart](../07-agent-mcp-client-streamablehttp/README.md).
* You can add resiliency, observability, and security using Dapr building blocks.
* To see multi-tool usage and structured output patterns, refer to the [LLM Call quickstart](../02-llm-call-open-ai/README.md).

---

# 2. Agent with Durable Execution

This example converts the previous agent into a durable agent that can resume after interruption.

```bash
dapr run --app-id durable-agent --resources-path resources -- python 02_agent_with_durable_execution.py
```

On a different terminal, trigger the agent:

```bash
curl -i -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the weather in London?"}'
```
You'll receive a WORKFLOW_ID in response. 

Query agent response using the WORKFLOW_ID:

```bash
curl -i -X GET http://localhost:8001/run/WORKFLOW_ID
```

Replace `WORKFLOW_ID` with the ID returned from the POST request.

## Expected Behavior

* The agent exposes a REST endpoint on port 8001 and is ready to accept new prompts.
* You send a prompt to the agent and receive back a workflow ID representing the durable execution.
* You query the workflow using that ID and see the final outcome of the prompt, even if the agent was stopped and restarted during execution.

**Test durability:**

To validate the durability of a `DurableAgent`, you can interrupt its execution while it is running and observe that the workflow continues from the exact point it stopped once the agent is restarted. This demonstrates that durability is handled by the workflow engine rather than the running process. To reproduce this behavior, follow these steps:

1. Send another prompt to the agent using the curl command above.
2. While it is executing (the tool call intentionally waits 5 seconds to give you time), kill the agent by pressing Ctrl+C.
3. Bring the agent back by running the same dapr run command again.
4. You will see that the agent continues from the point where it left off until the workflow completes.
5. Verify the completion using the workflow ID with the GET command above.

The durability comes from the underlying workflow execution state rather than the in-memory process. This ensures the agent interaction is not lost mid-execution, and the agent does not need to repeat earlier LLM calls or tool operations. The workflow engine restores the previous state and continues forward deterministically.

## How to Extend This Example

* Add custom workflow activities for business logic or integrations.
* Combine multiple agents inside the same durable workflow.
* To see multi-step workflows with LLM interactions, refer to the [LLM-based workflows quickstart](../04-llm-based-workflows/README.md).

---

# 3. Workflow with LLM Activities

This example shows how a Dapr workflow performs LLM interactions in a deterministic sequence. It illustrates that LLM integration inside workflows remains simple and requires no additional complexity.

```bash
dapr run --app-id workflow-llms --resources-path resources -- python 03_workflow_with_llms.py
```

## Expected Behavior

The workflow plans a blog post by generating an outline, then creates the final post from that plan.

## How This Works

1. The workflow calls the LLM in a fixed sequence which keeps execution deterministic.
2. Each step is persisted by the workflow engine.
3. The workflow can be restarted without losing progress.

## How to Extend This Example

* Add custom activities that mix business logic and LLM steps.
* Use structured output to enforce schema based LLM responses.
* To see structured output and validation, refer to the [LLM Call quickstart](../02-llm-call-open-ai/README.md).


---

# 4. Workflow with Agent Activities

This example demonstrates orchestrating multiple agents through a deterministic workflow. Each agent has tools and memory, and while the agents are not durable, the workflow coordinates them within a durable execution that ensures reliable execution.

```bash
dapr run --app-id workflow-agents --resources-path resources -- python 04_workflow_with_agents.py
```

## Expected Behavior

The workflow orchestrates two agents in sequence. The triage agent fetches customer information using a tool, and the expert agent uses that information to produce recommendations. The workflow ensures these interactions run reliably within a durable process.

## How This Works

1. The workflow invokes each agent as an activity.
2. The workflow passes inputs to the agents and collects structured outputs.
3. If an agent fails or restarts the workflow retries or resumes because workflow state is persisted.
4. The agents step themselves are not durable but the workflow guarantees durable orchestration and agent execution.

## How to Extend This Example
You can create workflows that combine regular workflow activities with custom actions, LLM interactions, and agent calls. All of these can be orchestrated deterministically and reliably within the same workflow.

---

# Other Multi-Agent Examples
If you want to coordinate multiple agents that run in separate applications or communicate through Pub/Sub, check out the [multi-agent workflows quickstart](../05-multi-agent-workflows/README.md).

---

# Troubleshooting

1. **API Key Issues**: If you see an authentication error, verify your OpenAI API key in the `.env` file
2. **Python Version**: If you encounter compatibility issues, make sure you're using Python 3.10+
3. **Environment Activation**: Ensure your virtual environment is activated before running examples
4. **Import Errors**: If you see module not found errors, verify that `pip install -r requirements.txt` completed successfully

# Next Steps

Learn how to use structured outputs with LLMs in the [LLM Call quickstart](../02-llm-call-open-ai/README.md).
 