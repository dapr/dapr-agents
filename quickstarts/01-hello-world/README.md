# Hello World with Dapr Agents

This quickstart provides a hands-on introduction to Dapr Agents through simple examples. You'll learn the fundamentals of working with LLMs, creating basic agents, implementing the ReAct pattern, and setting up simple workflows - all in less than 20 lines of code per example.

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
OPENAI_BASE_URL="https://api.openai.com/v1"
```

Replace `your_api_key_here` with your actual OpenAI API key.

## Examples

### 1. Basic LLM Usage

Run the basic LLM example to see how to interact with OpenAI's language models:

```bash
python 01_ask_llm.py
```

This example demonstrates the simplest way to use Dapr Agents' OpenAIChatClient:

```python
from floki import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()
llm = OpenAIChatClient()
response = llm.generate("Tell me a joke")
print(response.get_content())
```

**Expected output:** The LLM will respond with a joke.

### 2. Simple Agent with Tools

Run the agent example to see how to create an agent with custom tools:

```bash
python 02_build_agent.py
```

This example shows how to create a basic agent with a custom tool:

```python
from floki import tool, Agent

@tool
def weather() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"

weather_agent = Agent(
    name="WeatherAgent",
    role="Weather Assistant",
    instructions=["Help users with weather information"],
    tools=[weather]
)

response = weather_agent.run("What's the weather?")
print(response)
```

**Expected output:** The agent will use the weather tool to provide the current weather.

### 3. ReAct Pattern Implementation

Run the ReAct pattern example to see how to create an agent that can reason and act:

```bash
python 03_reason_act.py
```

This example demonstrates the ReAct pattern with multiple tools:

```python
from floki import tool, ReActAgent

@tool
def search_weather(city: str) -> str:
    """Get weather information for a city."""
    weather_data = {"london": "Rainy", "paris": "Sunny"}
    return weather_data.get(city.lower(), "Unknown")

@tool
def get_activities(weather: str) -> str:
    """Get activity recommendations."""
    activities = {"Rainy": "Visit museums", "Sunny": "Go hiking"}
    return activities.get(weather, "Stay comfortable")

react_agent = ReActAgent(
    name="TravelAgent",
    role="Travel Assistant",
    instructions=["Check weather, then suggest activities"],
    tools=[search_weather, get_activities]
)

react_agent.run("What should I do in London today?")
```

**Expected output:** The agent will first check the weather in London, find it's rainy, and then recommend visiting museums.

### 4. Simple Workflow

Run the workflow example to see how to create a multi-step LLM process:

```bash
python 04_chain_tasks.py
```

This example demonstrates how to create a workflow with multiple tasks:

```python
from floki import WorkflowApp
from floki.types import DaprWorkflowContext

wfapp = WorkflowApp()

@wfapp.workflow(name='research_workflow')
def analyze_topic(ctx: DaprWorkflowContext, topic: str):
    outline = yield ctx.call_activity(create_outline, input=topic)
    blog_post = yield ctx.call_activity(write_blog, input=outline)
    return blog_post

@wfapp.task(description="Create a detailed outline about {topic}")
def create_outline(topic: str) -> str:
    pass

@wfapp.task(description="Write a comprehensive blog post following this outline: {outline}")
def write_blog(outline: str) -> str:
    pass

if __name__ == '__main__':
    results = wfapp.run_and_monitor_workflow(
        workflow=analyze_topic,
        input="AI Agents"
    )
```

**Expected output:** The workflow will create an outline about AI Agents and then generate a blog post based on that outline.

## Key Concepts

- **OpenAIChatClient**: The interface for interacting with OpenAI's LLMs
- **Agent**: A class that combines an LLM with tools and instructions
- **@tool decorator**: A way to create tools that agents can use
- **ReActAgent**: An agent that follows the Reasoning + Action pattern
- **WorkflowApp**: A Dapr-powered way to create stateful, multi-step processes

## Dapr Integration

While these simple examples run without Dapr services, they're built on Dapr Agents which provides:

- **Resilience**: Dapr's state management for durable workflows
- **Scalability**: Services can be distributed across infrastructure
- **Interoperability**: Components integrate with various backend systems

In the later quickstarts, you'll see explicit Dapr integration through state stores, pub/sub, and actor services.

## Troubleshooting

1. **API Key Issues**: If you see an authentication error, verify your OpenAI API key in the `.env` file
2. **Python Version**: If you encounter compatibility issues, make sure you're using Python 3.10+
3. **Environment Activation**: Ensure your virtual environment is activated before running examples
4. **Import Errors**: If you see module not found errors, verify that `pip install -r requirements.txt` completed successfully

## Next Steps

After completing these examples, move on to the [LLM Call quickstart](../02-llm-call) to learn more about structured outputs from LLMs.