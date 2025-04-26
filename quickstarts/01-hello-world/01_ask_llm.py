from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()
llm = OpenAIChatClient()
response = llm.generate("Tell me a joke")
if len(response.get_content()) > 0:
    print("Got response:", response.get_content())
