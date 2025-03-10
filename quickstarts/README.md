# Dapr Agents Quickstarts

A collection of example projects demonstrating how to use Dapr Agents to build applications with LLM-powered autonomous agents and event-driven workflows. Each quickstart builds upon the previous one, introducing new concepts incrementally.

## Prerequisites

To run these quickstarts, you'll need:
- Python 3.10 or higher
- An OpenAI API key
- Dapr CLI and Docker (for workflow examples)

## Available Quickstarts

### 01 - Hello World

A rapid introduction to Dapr Agents core concepts through simple examples:

- **Basic LLM Usage**: Simple text generation with OpenAI models
- **Creating Agents**: Building agents with custom tools in under 20 lines of code
- **ReAct Pattern**: Implementing reasoning and action cycles
- **Simple Workflows**: Setting up multi-step LLM processes

[Go to Hello World](./01-hello-world)

### 02 - LLM Call

Learn how to interact with Language Models using Dapr Agents:

- **Text Completion**: Generating responses to prompts
- **Structured Outputs**: Converting LLM responses to Pydantic objects

This quickstart shows both basic text generation and structured data extraction from LLMs.

[Go to LLM Call](./02_llm_call_open_ai)

### 03 - Agent Tool Call

Create your first AI agent with custom tools:

- **Tool Definition**: Creating reusable tools with the @tool decorator
- **Agent Configuration**: Setting up agents with roles, goals, and tools
- **Function Calling**: Enabling LLMs to execute Python functions

This quickstart demonstrates how to build a weather assistant that can fetch information and perform actions.

[Go to Agent Tool Call](./03-agent-tool-call)

### 04 - Agentic Workflow

Introduction to Dapr workflows with Dapr Agents:

- **Workflow Basics**: Understanding Dapr's workflow capabilities
- **Task Chaining**: Creating resilient multi-step processes
- **LLM-powered Tasks**: Using language models in workflows
- **Comparison**: Seeing the difference between pure Dapr and Dapr Agents approaches

This quickstart shows how to orchestrate multi-step processes that combine deterministic tasks with LLM-powered reasoning.

[Go to Agentic Workflow](./04-agentic-workflow)

### 05 - Multi-Agent Workflows

Advanced example of event-driven workflows with multiple autonomous agents:

- **Multi-agent Systems**: Creating a network of specialized agents
- **Event-driven Architecture**: Implementing pub/sub messaging between agents
- **Actor Model**: Using Dapr Actors for stateful agent management
- **Workflow Orchestration**: Coordinating agents through different selection strategies

This quickstart demonstrates a Lord of the Rings themed multi-agent system where agents collaborate to solve problems.

[Go to Multi-Agent Workflows](./05-multi-agent-workflow)

## Getting Started

1. Clone this repository
```bash
git clone https://github.com/dapr-sandbox/dapr-agents/
cd dapr-agents/quickstarts
```

2. Set up environment variables
```bash
# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. For workflow examples, initialize Dapr
```bash
dapr init
```

4. Choose a quickstart and follow its specific README
