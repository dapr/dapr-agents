# Third-Party Tools Quickstart

This quickstart demonstrates how to integrate third-party tools from other agentic frameworks like LangChain and CrewAI with dapr-agents. You'll learn how to import and use existing tools from these ecosystems to work with Dapr Agents.

## Prerequisites

- Python 3.10 (recommended)
- pip package manager
- OpenAI API key
- For LangChain example: langchain, langchain-community
- For CrewAI example: crewai, crewai-tools

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

### LangChain Tools Integration

This example shows how to integrate LangChain tools with dapr-agents:

1. In `agent_with_langchain_tool.py`, we integrate DuckDuckGo search with a custom word counter tool:

```python
from dotenv import load_dotenv
import asyncio
from langchain_community.tools import DuckDuckGoSearchRun
from dapr_agents.agent import Agent
from dapr_agents.tool import LangchainTool, tool
from pydantic import BaseModel, Field

load_dotenv()

# Create a search args model
class SearchArgs(BaseModel):
    query: str = Field(..., description="The search query to look up information for")

@tool
def count_words(text: str) -> str:
    """Count the number of words, lines, and characters in the text."""
    words = len(text.split())
    lines = len(text.splitlines())
    chars = len(text)
    return f"Text Statistics:\n- Words: {words}\n- Lines: {lines}\n- Characters: {chars}"

async def main():
    # Create LangChain tool - DuckDuckGo search
    ddg_search_tool = DuckDuckGoSearchRun()

    # Wrap LangChain tool with dapr's LangchainTool adapter
    search_tool = LangchainTool(
        ddg_search_tool,
        name="WebSearch",
        description="Search the web for current information on any topic"
    )
    
    # Set the args model for proper argument handling
    search_tool.args_model = SearchArgs

    # Create a dapr agent with both tools
    agent = Agent(
        """You are a helpful assistant that can search the web and analyze text.
        Use the WebSearch tool to find information about topics,
        then use the CountWords tool to analyze the text statistics of the results.""",
        tools=[search_tool, count_words]
    )

    # Run the agent with a query
    query = "What is Dapr and then count the words in your search results."
    print(f"User: {query}")
    
    # Properly await the run method
    result = await agent.run(query)
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Run the LangChain tools agent:

```bash
python agent_with_langchain_tool.py
```

### CrewAI Tools Integration

This example shows how to integrate CrewAI tools:

1. In `agent_with_crewai_tool.py`, we integrate a file reading tool with the same word counter:

```python
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from crewai_tools import FileReadTool
from dapr_agents.agent import Agent
from dapr_agents.tool import CrewAITool, tool

load_dotenv()

@tool
def count_words(text: str) -> str:
    """Count the number of words, lines, and characters in the text."""
    words = len(text.split())
    lines = len(text.splitlines())
    chars = len(text)
    return f"Text Statistics:\n- Words: {words}\n- Lines: {lines}\n- Characters: {chars}"

async def main():
    # Create CrewAI FileReadTool
    file_read_tool = FileReadTool(file_path="sample_data.txt")
    
    # Wrap with CrewAITool adapter
    file_tool = CrewAITool(
        file_read_tool,
        name="ReadFile",
        description="Reads text from the sample file"
    )

    # Create a dapr agent with both tools - CrewAI wrapped tool and native tool
    agent = Agent(
        """You are a helpful assistant that can read files and analyze text.
        The ReadFile tool is already configured to read from a specific sample file, 
        so you can just use it without arguments.
        After getting the content, you can use the CountWords tool to analyze the text statistics.""",
        tools=[file_tool, count_words]
    )

    # Run the agent with a query about the file
    query = "Please read the sample file and tell me its word count and other statistics."
    print(f"User: {query}")
    
    # Properly await the run method
    result = await agent.run(query)
    print(f"Agent: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Run the CrewAI tools agent:

```bash
python agent_with_crewai_tool.py
```

## Key Concepts

### Tool Adaptation
- Third-party tools need to be wrapped with adapter classes to work with dapr-agents
- `LangchainTool` adapter converts LangChain tools to the dapr-agents format
- `CrewAITool` adapter converts CrewAI tools to the dapr-agents format
- You can customize names and descriptions when adapting tools

### Agent Setup
- The `Agent` class works the same way as with native dapr-agents tools
- Adapted tools are provided in the tools list alongside native tools
- You can mix and match native dapr-agents tools with adapted third-party tools

### Execution Flow
1. The agent receives a user query
2. The LLM determines which tool(s) to use based on the query
3. The adapter converts the call format for the third-party tool
4. The third-party tool executes and returns results
5. The adapter converts the results back to the dapr-agents format
6. The final answer is provided to the user

## Popular Tool Options

You can use many different tools from these frameworks with the appropriate adapter. Check the [LangChain tools](https://python.langchain.com/docs/integrations/tools/) and [CrewAI tools](https://github.com/crewAIInc/crewAI-tools/) for more options.

## Troubleshooting

1. **OpenAI API Key**: Ensure your key is correctly set in the `.env` file
2. **Tool Dependencies**: Verify that you've installed all required packages for the specific tools
3. **Tool Execution Errors**: Check tool implementations for exceptions
4. **Adaptation Issues**: Make sure you're using the correct adapter for each tool type
5. **File Paths**: For file-based tools like CrewAI's FileReadTool, ensure the files exist at the specified path

## Next Steps

After completing this quickstart, explore how to combine native dapr-agents tools with third-party tools in more complex workflows. 