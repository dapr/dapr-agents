import json

from dapr_agents import OpenAIChatClient
from dapr_agents.types import UserMessage
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# Define our data model
class Dog(BaseModel):
    name: str
    breed: str
    reason: str


# Initialize the chat client
llm = OpenAIChatClient()

# Get structured response
response = llm.generate(
    messages=[UserMessage("One famous dog in history.")], response_format=Dog
)

print(json.dumps(response.model_dump(), indent=2))
