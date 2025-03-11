# Dapr Agents Quickstarts

[Quickstarts](https://github.com/dapr/dapr-agents/tree/main/quickstarts) demonstrate how to use Dapr Agents to build applications with LLM-powered autonomous agents and event-driven workflows. Each quickstart builds upon the previous one, introducing new concepts incrementally.


!!! info
    Not all quickstarts require Docker, but it is recommended to have your [local Dapr environment set up](../installation.md) with Docker for the best development experience and to follow the steps in this guide seamlessly.

## Quickstarts
 
| Scenario | What You’ll Learn |
| --- | --- |
| [01 - Hello World](https://github.com/dapr/dapr-agents/tree/main/quickstarts/01-hello-world) | - **Basic LLM Usage**: Simple text generation with OpenAI models <br> - **Creating Agents**: Building agents with custom tools in under 20 lines of code <br> - **ReAct Pattern**: Implementing reasoning and action cycles <br> - **Simple Workflows**: Setting up multi-step LLM processes |
| [02 - LLM Call](https://github.com/dapr/dapr-agents/tree/main/quickstarts/02_llm_call_open_ai) | - **Text Completion**: Generating responses to prompts <br> - **Structured Outputs**: Converting LLM responses to Pydantic objects <br> - Covers both basic text generation and structured data extraction from LLMs |
| [03 - Agent Tool Call](https://github.com/dapr/dapr-agents/tree/main/quickstarts/03-agent-tool-call) | - **Tool Definition**: Creating reusable tools with the `@tool` decorator <br> - **Agent Configuration**: Setting up agents with roles, goals, and tools <br> - **Function Calling**: Enabling LLMs to execute Python functions <br> - Demonstrates a weather assistant that fetches information and performs actions |
| [04 - Agentic Workflow](https://github.com/dapr/dapr-agents/tree/main/quickstarts/04-agentic-workflow) | - **Workflow Basics**: Understanding Dapr's workflow capabilities <br> - **Task Chaining**: Creating resilient multi-step processes <br> - **LLM-powered Tasks**: Using language models in workflows <br> - **Comparison**: Difference between pure Dapr and Dapr Agents approaches |
| [05 - Multi-Agent Workflows](https://github.com/dapr/dapr-agents/tree/main/quickstarts/05-multi-agent-workflow-dapr-workflows) | - **Multi-agent Systems**: Creating a network of specialized agents <br> - **Event-driven Architecture**: Implementing pub/sub messaging between agents <br> - **Actor Model**: Using Dapr Actors for stateful agent management <br> - **Workflow Orchestration**: Coordinating agents through different selection strategies |
