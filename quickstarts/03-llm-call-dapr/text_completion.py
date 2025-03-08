from dapr_agents.llm.dapr import DaprChatClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the chat client and call
llm = DaprChatClient()
response = llm.generate("Name a famous dog!")
print("Response: ", response.get_content())
