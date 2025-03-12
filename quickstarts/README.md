# Dapr Agents Quickstarts

A collection of examples demonstrating how to use Dapr Agents to build applications with LLM-powered autonomous agents and event-driven workflows. Each quickstart builds upon the previous one, introducing new concepts incrementally.

## Prerequisites

To run these quickstarts, you'll need:
- [Python 3.10 or higher](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)
- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/)
- [OpenAI API Key](https://platform.openai.com/api-keys) (Used for tutorials, other LLMs are available too)


## Getting Started

1. Clone this repository
```bash
git clone https://github.com/dapr/dapr-agents/
cd dapr-agents/quickstarts
```

2. For workflow examples, [install Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) and initialize Dapr
```bash
dapr init
```

3. Choose a quickstart from the list below. Or click [here](./01-hello-world) to start with Hello-World.

## Available Quickstarts

### Hello World

A rapid introduction to Dapr Agents core concepts through simple demonstrations:

- **Basic LLM Usage**: Simple text generation with OpenAI models
- **Creating Agents**: Building agents with custom tools in under 20 lines of code
- **ReAct Pattern**: Implementing reasoning and action cycles in agents
- **Simple Workflows**: Setting up multi-step LLM processes

[Go to Hello World](./01-hello-world)

### LLM Call with Dapr Chat Client

Learn how to interact with Language Models using Dapr Agents' `DaprChatClient`:

- **Text Completion**: Generating responses to prompts
- **Swapping LLM providers**: switching LLM backends without application code change
- **Resilience**: Setting timeout, retry and circuit-breaking
- **PII Obfuscation** – Automatically detect and mask sensitive user information.


This quickstart shows basic text generation using plain text prompts and templates. Using the `DaprChatClient` you can target different LLM providers without changing your agent's code.

[Go to Dapr LLM Call](./02_llm_call_dapr)

### LLM Call with OpenAI Client

Learn how to interact with Language Models using Dapr Agents and native LLM client libraries.

- **Text Completion**: Generating responses to prompts
- **Structured Outputs**: Converting LLM responses to Pydantic objects

This quickstart shows both basic text generation and structured data extraction from LLMs. This quickstart uses the OpenAIChatClient which allows you to use audio and perform embeddings in addition to chat completion. 

*Note: Other quickstarts for specific clients are available for [Elevenlabs](./02_llm_call_elevenlabs), [Hugging Face](./02_llm_call_hugging_face), and [Nvidia](./02_llm_call_nvidia).*


[Go to OpenAI LLM call](./02_llm_call_open_ai)

### Agent Tool Call

Create your first AI agent with custom tools:

- **Tool Definition**: Creating reusable tools with the @tool decorator
- **Agent Configuration**: Setting up agents with roles, goals, and tools
- **Function Calling**: Enabling LLMs to execute Python functions

This quickstart demonstrates how to build a weather assistant that can fetch information and perform actions.

[Go to Agent Tool Call](./03-agent-tool-call)

### Agentic Workflow

Introduction to using stateful workflows with Dapr Agents:

- **LLM-powered Tasks**: Using language models in workflows
- **Task Chaining**: Creating resilient multi-step processes executing in sequence
- **Fan-out/Fan-in**:Executing activities in parallel; then synchronize these activities until all preceding activities have completed.


This quickstart demonstrates how to orchestrate sequential and parallel tasks using Dapr Agents' workflow capabilities.

[Go to Agentic Workflow](./04-agentic-workflow)

### Multi-Agent Workflows

Advanced example of event-driven workflows with multiple autonomous agents:

- **Multi-agent Systems**: Creating a network of specialized agents
- **Event-driven Architecture**: Implementing pub/sub messaging between agents
- **Actor Model**: Using Dapr Actors for stateful agent management
- **Workflow Orchestration**: Coordinating agents through different selection strategies

This quickstart demonstrates a Lord of the Rings themed multi-agent system where agents collaborate to solve problems.

*Note: To see Actor-based workflow see [Multi-Agent Actors](./05-multi-agent-workflow-actors).*

[Go to Multi-Agent Workflows](./05-multi-agent-workflow-dapr-workflows)
