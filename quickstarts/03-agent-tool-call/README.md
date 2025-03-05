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
from weather_tools import tools
from dapr_agents import Agent
from dotenv import load_dotenv

load_dotenv()

AIAgent = Agent(
    name = "Stevie",
    role = "Weather Assistant",
    goal = "Assist Humans with weather related tasks.",
    instructions = ["Get accurate weather information", "From time to time, you can also Jump after answering the weather question."],
    tools=tools
)

AIAgent.run("What is the weather in Virginia, New York and Washington DC?")
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

- **@tool decorator**: Transforms Python functions into agent-accessible tools
- **Pydantic models**: Define the schema and validate arguments for tools
- **Agent**: Combines an LLM with tools and instructions
- **Function Calling**: The LLM intelligently decides when and how to call tools
- **Agent Memory**: Conversation history that the agent can reference

## Understanding the Code

### Tool Definition
- The `@tool` decorator registers functions as tools with the agent
- Each tool has a docstring that helps the LLM understand its purpose
- Pydantic models provide type-safety for tool arguments

### Agent Setup
- The `Agent` class sets up a tool-calling agent by default
- The `role`, `goal`, and `instructions` guide the agent's behavior
- Tools are provided as a list for the agent to use

### Execution Flow
1. The agent receives a user query
2. The LLM determines which tool(s) to use based on the query
3. The agent executes the tool with appropriate arguments
4. The results are returned to the LLM to formulate a response
5. The final answer is provided to the user

## Dapr Integration

Agents can be deployed as stateful, distributed services using Dapr:

- **State Management**: Agent memory can persist across restarts
- **Pub/Sub**: Agents can communicate through message buses
- **Actor Model**: Dapr Actors enable stateful agent components

## Working with Agent Memory

You can access and manage the agent's conversation history:

```python
# Access conversation history
AIAgent.chat_history

# Reset the agent's memory
AIAgent.reset_memory()
```

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Tool Execution Errors**: Check tool function implementations for exceptions
3. **Module Import Errors**: Verify that requirements are installed correctly

## Next Steps

After completing this quickstart, move on to the [Agentic Workflow quickstart](../04-agentic-workflow) to learn how to orchestrate multi-step processes combining deterministic tasks with LLM-powered reasoning.