from dotenv import load_dotenv

from dapr_agents import NVIDIAChatClient
from dapr_agents.types import LLMChatResponse, UserMessage

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = NVIDIAChatClient()
response: LLMChatResponse = llm.generate("Name a famous dog!")

if response.get_message() is not None:
    print("Response: ", response.get_message().content)

# Chat completion using a prompty file for context
llm = NVIDIAChatClient.from_prompty("basic.prompty")
response: LLMChatResponse = llm.generate(input_data={"question": "What is your name?"})

if response.get_message() is not None:
    print("Response with prompty: ", response.get_message().content)

# Chat completion with user input
llm = NVIDIAChatClient()
response: LLMChatResponse = llm.generate(messages=[UserMessage("hello")])

if (
    response.get_message() is not None
    and "hello" in response.get_message().content.lower()
):
    print("Response with user input: ", response.get_message().content)
