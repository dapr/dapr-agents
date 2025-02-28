from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()
llm = OpenAIChatClient()
response = llm.generate("Tell me a joke")
print(response.get_content())