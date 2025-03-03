from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the chat client and call
llm = OpenAIChatClient()
response = llm.generate("Name a famous dog!")

print(response.get_content())