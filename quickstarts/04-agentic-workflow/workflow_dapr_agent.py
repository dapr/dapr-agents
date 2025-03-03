from dapr_agents import WorkflowApp
from dapr_agents.types import DaprWorkflowContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the WorkflowApp
wfapp = WorkflowApp()

# Define Workflow logic
@wfapp.workflow(name='lotr_workflow')
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2

@wfapp.task(description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name only
""")
def get_character() -> str:
    pass

@wfapp.task(description="What is a famous line by {character}",)
def get_line(character: str) -> str:
    pass

if __name__ == '__main__':
    results = wfapp.run_and_monitor_workflow(task_chain_workflow)
    print(f"Famous Line: {results}")