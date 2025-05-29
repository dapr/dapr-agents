# Agent Tool Call with Dapr Agents

This quickstart demonstrates how to create an AI agent with custom tools using Dapr Agents. You'll learn how to build a weather assistant that can fetch information and perform actions using defined tools through LLM-powered function calls.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key

## Environment Setup

```bash
# Create a virtual environment
python3.10 -m venv .venv

# Activate the virtual environment 
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Examples

### Tool Creation and Agent Execution

This example shows how to create tools and an agent that can use them:

1. First, create the tools in `weather_tools.py`:

```python
from dapr_agents import tool
from pydantic import BaseModel, Field

class GetWeatherSchema(BaseModel):
    location: str = Field(description="location to get weather for")

@tool(args_model=GetWeatherSchema)
def get_weather(location: str) -> str:
    """Get weather information based on location."""
    import random
    temperature = random.randint(60, 80)
    return f"{location}: {temperature}F."

class JumpSchema(BaseModel):
    distance: str = Field(description="Distance for agent to jump")

@tool(args_model=JumpSchema)
def jump(distance: str) -> str:
    """Jump a specific distance."""
    return f"I jumped the following distance {distance}"

tools = [get_weather, jump]
```

2. Then, create the agent in `weather_agent.py`:

```python
import asyncio
from weather_tools import tools
from dapr_agents import Agent
from dotenv import load_dotenv

load_dotenv()

AIAgent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Get accurate weather information",
        "From time to time, you can also jump after answering the weather question."
    ],
    tools=tools
)

# Wrap your async call
async def main():
    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")

if __name__ == "__main__":
    asyncio.run(main())
```

3. Run the weather agent:

<!-- STEP
name: Run text completion example
expected_stdout_lines:
  - "user:"
  - "What is the weather in Virginia, New York and Washington DC?"
  - "assistant:"
  - "Function name: GetWeather (Call Id:"
  - 'Arguments: {"location":'
  - "assistant:"
  - "Function name: GetWeather (Call Id:"
  - 'Arguments: {"location":'
  - "assistant:"
  - "Function name: GetWeather (Call Id:"
  - 'Arguments: {"location":'
  - "GetWeather(tool)"
  - "Virginia"
  - "GetWeather(tool)"
  - "New York"
  - "GetWeather(tool)"
  - "Washington DC"
timeout_seconds: 30
output_match_mode: substring
-->
```bash
python weather_agent.py
```
<!-- END_STEP -->

**Expected output:** The agent will identify the locations and use the get_weather tool to fetch weather information for each city.

## Key Concepts

### Tool Definition
- The `@tool` decorator registers functions as tools with the agent
- Each tool has a docstring that helps the LLM understand its purpose
- Pydantic models provide type-safety for tool arguments

### Agent Setup
- The `Agent` class sets up a tool-calling agent by default
- The `role`, `goal`, and `instructions` guide the agent's behavior
- Tools are provided as a list for the agent to use
- Agent Memory keeps the conversation history that the agent can reference


### Execution Flow
1. The agent receives a user query
2. The LLM determines which tool(s) to use based on the query
3. The agent executes the tool with appropriate arguments
4. The results are returned to the LLM to formulate a response
5. The final answer is provided to the user

## Working with Agent Memory

You can access and manage the agent's conversation history too. Add this code fragment to the end of `weather_agent.py` and run it again.

```python
# View the history after first interaction
print("Chat history after first interaction:")
print(AIAgent.chat_history)

# Second interaction (agent will remember the first one)
await AIAgent.run("How about in Seattle?")

# View updated history
print("Chat history after second interaction:")
print(AIAgent.chat_history)

# Reset memory
AIAgent.reset_memory()
print("Chat history after reset:")
print(AIAgent.chat_history)  # Should be empty now
```
This will show agent interaction history growth and reset.

### Persistent Agent Memory

Dapr Agents allows for agents to retain long-term memory by providing automatic state management of the history. The state can be saved into a wide variety of [Dapr supported state stores](https://docs.dapr.io/reference/components-reference/supported-state-stores/).

To configure persistent agent memory, follow these steps:

1. Set up the state store configuration. Here's an example of working with local Redis:

```yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: historystore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: localhost:6379
  - name: redisPassword
    value: ""
```

Save the file in a `./components` dir.

2. Enable Dapr memory in code

```python
import asyncio
from weather_tools import tools
from dapr_agents import Agent
from dotenv import load_dotenv
from dapr_agents.memory import ConversationDaprStateMemory

load_dotenv()

AIAgent = Agent(
    name="Stevie",
    role="Weather Assistant",
    goal="Assist Humans with weather related tasks.",
    instructions=[
        "Get accurate weather information",
        "From time to time, you can also jump after answering the weather question."
    ],
    memory=ConversationDaprStateMemory(store_name="historystore", session_id="some-id"),
    tools=tools
)

# Wrap your async call
async def main():
    await AIAgent.run("What is the weather in Virginia, New York and Washington DC?")

if __name__ == "__main__":
    asyncio.run(main())
```

3. Run the agent with Dapr

```bash
dapr run --app-id weatheragent --resources-path ./components -- python weather_agent_dapr.py
```

## Available Agent Types

Dapr Agents provides several agent implementations, each designed for different use cases:

### 1. Standard Agent (ToolCallAgent)
The default agent type, designed for tool execution and straightforward interactions. It receives your input, determines which tools to use, executes them directly, and provides the final answer. The reasoning process is mostly hidden from you, focusing instead on delivering concise responses.

### 2. ReActAgent
Implements the Reasoning-Action framework for more complex problem-solving with explicit thought processes.
When you interact with it, you'll see explicit "Thought", "Action", and "Observation" steps as it works through your request, providing transparency into how it reaches conclusions.

### 3. OpenAPIReActAgent
There is one more agent that we didn't run in this quickstart. OpenAPIReActAgent specialized agent for working with OpenAPI specifications and API integrations. When you ask about working with an API, it will methodically identify relevant endpoints, construct proper requests with parameters, handle authentication, and execute API calls on your behalf.

```python
from dapr_agents import Agent
from dapr_agents.tool.utils.openapi import OpenAPISpecParser
from dapr_agents.storage import VectorStore

# This agent type requires additional components
openapi_agent = Agent(
    name="APIExpert",
    role="API Expert",
    pattern="openapireact",  # Specify OpenAPIReAct pattern
    spec_parser=OpenAPISpecParser(),
    api_vector_store=VectorStore(),
    auth_header={"Authorization": "Bearer token"}
)
```

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Tool Execution Errors**: Check tool function implementations for exceptions
3. **Module Import Errors**: Verify that requirements are installed correctly

## Next Steps

After completing this quickstart, move on to the [Agentic Workflow quickstart](../04-agentic-workflow/README.md) to learn how to orchestrate multi-step processes combining deterministic tasks with LLM-powered reasoning.