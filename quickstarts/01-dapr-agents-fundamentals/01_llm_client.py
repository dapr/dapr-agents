from dapr_agents.llm import DaprChatClient
from dapr_agents.types import LLMChatResponse

# Basic chat completion
llm = DaprChatClient(component_name="llm-provider")
response: LLMChatResponse = llm.generate(
    "Guess what is the weather in London right now!"
)

print("Response: ", response.get_message().content)
