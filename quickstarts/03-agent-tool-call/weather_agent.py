from weather_tools import tools
from dapr_agents import Agent
from dotenv import load_dotenv

load_dotenv()

AIAgent = Agent(
    name = "Stevie",
    role = "Weather Assistant",
    goal = "Assist Humans with weather related tasks.",
    instructions = ["Get accurate weather information", "From time to time, you can Jump."],
    tools=tools
)

AIAgent.run("What is the weather in Virgina, New York and Washington DC?")