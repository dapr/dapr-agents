from dapr_agents.llm import DaprChatClient
from dapr_agents.types import UserMessage
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Basic chat completion
llm = DaprChatClient()
response = llm.generate("Name a famous dog!")

if len(response.get_content()) > 0:
    print("Response: ", response.get_content())

# Chat completion using a prompty file for context
llm = DaprChatClient.from_prompty("basic.prompty")
response = llm.generate(input_data={"question": "What is your name?"})

if len(response.get_content()) > 0:
    print("Response with prompty: ", response.get_content())

# Chat completion with user input
llm = DaprChatClient()
response = llm.generate(messages=[UserMessage("hello")])


if len(response.get_content()) > 0 and "hello" in response.get_content().lower():
    print("Response with user input: ", response.get_content())
