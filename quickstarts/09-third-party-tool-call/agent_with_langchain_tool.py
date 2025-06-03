#!/usr/bin/env python3
import os
import asyncio
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun

from dapr_agents.agent import Agent
from dapr_agents.tool import tool
from dapr_agents.tool.third_party import LangchainTool
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
    return (
        f"Text Statistics:\n- Words: {words}\n- Lines: {lines}\n- Characters: {chars}"
    )


async def main():
    # Create LangChain tool - DuckDuckGo search
    ddg_search_tool = DuckDuckGoSearchRun()

    # Wrap LangChain tool with dapr's LangchainTool adapter
    search_tool = LangchainTool(
        ddg_search_tool,
        name="WebSearch",
        description="Search the web for current information on any topic",
    )

    # Set the args model for proper argument handling
    search_tool.args_model = SearchArgs

    # Create a dapr agent with both tools
    agent = Agent(
        """You are a helpful assistant that can search the web and analyze text.
        Use the WebSearch tool to find information about topics,
        then use the CountWords tool to analyze the text statistics of the results.""",
        tools=[search_tool, count_words],
    )

    # Run the agent with a query
    query = "What is Dapr and then count the words in your search results."
    print(f"User: {query}")

    # Properly await the run method
    result = await agent.run(query)
    print(f"Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
