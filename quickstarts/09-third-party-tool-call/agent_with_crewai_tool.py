#!/usr/bin/env python3

import os
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
    return (
        f"Text Statistics:\n- Words: {words}\n- Lines: {lines}\n- Characters: {chars}"
    )


async def main():
    # Create CrewAI FileReadTool
    file_read_tool = FileReadTool(file_path="sample_data.txt")

    # Wrap with CrewAITool adapter
    file_tool = CrewAITool(
        file_read_tool, name="ReadFile", description="Reads text from the sample file"
    )

    # Create a dapr agent with both tools - CrewAI wrapped tool and native tool
    agent = Agent(
        """You are a helpful assistant that can read files and analyze text.
        The ReadFile tool is already configured to read from a specific sample file, 
        so you can just use it without arguments.
        After getting the content, you can use the CountWords tool to analyze the text statistics.""",
        tools=[file_tool, count_words],
    )

    # Run the agent with a query about the file
    query = (
        "Please read the sample file and tell me its word count and other statistics."
    )
    print(f"User: {query}")

    # Properly await the run method
    result = await agent.run(query)
    print(f"Agent: {result}")


if __name__ == "__main__":
    asyncio.run(main())
