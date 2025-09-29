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
- **Tool Definition**: Custom tools with the @tool decorator
- **Function Calling**: Enabling LLMs to execute Python functions

This quickstart demonstrates how to build a weather assistant that can fetch information and perform actions.

[Go to Agent Tool Call](./03-agent-tool-call)

### Durable Agent Tool Call (with Dapr Workflows)

Create a stateful AI agent using Dapr Agents' DurableAgent and Dapr workflows:

* **Durable Agent**: Maintains state across runs for reliable, long-running tasks
* **Workflow Integration**: Uses Dapr workflows for persistence and recovery
* **Tool Definition**: Custom tools with the @tool decorator
* **Function Calling**: LLM-powered Python function execution

This quickstart demonstrates how to build a weather assistant with durable, workflow-enabled capabilities.

[Go to Durable Agent Tool Call](./03-durable-agent-tool-call/)

### LLM-based Workflow Patterns

Learn to orchestrate stateful, resilient workflows powered by Language Models (LLMs) using Dapr Agents:

- **LLM-powered Tasks**: Automate reasoning and decision-making in workflows
- **Task Chaining**: Build multi-step processes with reliable state management
- **Fan-out/Fan-in**: Execute activities in parallel and synchronize results for robust automation

This quickstart demonstrates how to design and run sequential and parallel workflows using Dapr Agents and LLMs for advanced orchestration.

[Go to LLM-based Workflow Patterns](./04-llm-based-workflows/)

### Multi-Agent Workflows

Advanced example of event-driven workflows with multiple autonomous agents:

- **Multi-agent Systems**: Creating a network of specialized agents
- **Event-driven Architecture**: Implementing pub/sub messaging between agents
- **Actor Model**: Using Dapr Actors for stateful agent management
- **Workflow Orchestration**: Coordinating agents through different selection strategies

This quickstart demonstrates a Lord of the Rings themed multi-agent system where agents collaborate to solve problems.


[Go to Multi-Agent Workflows](./05-multi-agent-workflows)

### Conversational Document Agent with Chainlit

Build a fully functional, agent that parses unstructured documents, learns them, and enables users to chat over their contents with persistent conversational memory. Integrates Dapr with Chainlit for a ready-to-use chat UI.

- **Converse With Unstructured Data**: Upload documents, parse, contextualize, and chat with them
- **Conversational Memory**: Agent maintains context across interactions in your database of choice
- **UI Interface**: Out-of-the-box, LLM-ready chat interface using Chainlit
- **Cloud Agnostic**: File uploads handled by Dapr, configurable for different backends

This quickstart demonstrates how to build a document agent with memory and chat capabilities, using Dapr and Chainlit.


[Go to Document Agent with Chainlit](./06-document-agent-chainlit)

### MCP Agent Quickstarts

Explore agents that use the Model Context Protocol (MCP) to connect to tools via different transports. These quickstarts show how to build agents that leverage MCP for distributed, modular, and cloud-native tool calling.

#### MCP Agent with SSE Transport

Build an agent that connects to MCP tools over Server-Sent Events (SSE) for network-based, distributed communication.

- **MCP Tool Definition**: Register Python functions as MCP tools
- **SSE Transport**: Connect agents and tools over HTTP SSE endpoints
- **Dapr Integration**: DurableAgent runs as a Dapr service with state and pubsub

This quickstart demonstrates how to expose tools via SSE and connect agents to them for real-time, distributed workflows.

[Go to MCP Agent with SSE Transport](./07-agent-mcp-client-sse)

#### MCP Agent with STDIO Transport

Build an agent that connects to MCP tools via STDIO for local, process-based communication.

- **MCP Tool Definition**: Register Python functions as MCP tools
- **STDIO Transport**: Connect agents and tools in the same process using standard input/output
- **Simple Local Setup**: No network required, ideal for development and testing

This quickstart demonstrates how to expose tools via STDIO and connect agents to them for fast, local workflows.

[Go to MCP Agent with STDIO Transport](./07-agent-mcp-client-stdio)

#### MCP Agent with Streamable HTTP Transport

Build an agent that connects to MCP tools over Streamable HTTP for robust, cloud-native communication and streaming responses.

- **MCP Tool Definition**: Register Python functions as MCP tools
- **Streamable HTTP Transport**: Use HTTP(S) for requests and real-time streaming responses
- **Cloud-Native**: Ideal for distributed, containerized, or Kubernetes deployments

This quickstart demonstrates how to expose tools via Streamable HTTP and connect agents to them for scalable, modern workflows.


[Go to MCP Agent with Streamable HTTP Transport](./07-agent-mcp-client-streamablehttp)

### Conversational Agent over Postgres with MCP

Build a fully functional, agent that lets users ask any question of their Postgres database in natural language and get both results and structured analysis. This quickstart shows how to use MCP in Dapr Agents to connect to a database and provides a ChatGPT-like interface using Chainlit.

- **Conversational Knowledge Base**: Talk to your database in natural language, ask complex questions, and perform advanced analysis
- **Conversational Memory**: Agent maintains context across interactions in your database of choice
- **UI Interface**: Out-of-the-box, LLM-ready chat interface using Chainlit
- **Boilerplate-Free DB Layer**: MCP allows Dapr Agent to connect to the database without Postgres-specific code

This quickstart demonstrates how to build a database agent with memory and chat capabilities, using Dapr, MCP, and Chainlit.

[Go to Conversational Agent over Postgres with MCP](./08-data-agent-mcp-chainlit)

### Contributing to Dapr Agents Quickstarts

Please refer to our [Dapr Community Code of Conduct](https://github.com/dapr/community/blob/master/CODE-OF-CONDUCT.md)

For development setup and guidelines, see our [Development Guide](../docs/development/README.md#contributing-to-dapr-agents-quickstarts).
