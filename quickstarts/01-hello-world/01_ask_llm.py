from dotenv import load_dotenv

from dapr_agents import OpenAIChatClient
from dapr_agents.types.message import LLMChatResponse

# load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat client
llm = OpenAIChatClient()

# Generate a response from the LLM
response: LLMChatResponse = llm.generate("Tell me a joke")

# Print the Message content if it exists
if response.get_message() is not None:
    content = response.get_message().content
    print("Got response:", content)
