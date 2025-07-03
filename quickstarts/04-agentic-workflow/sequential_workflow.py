from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv

# # Uncomment this to use the enhanced version with DaprChatClient
# from dapr_agents.llm.dapr import DaprChatClient

# Load environment variables
load_dotenv()


# Define Workflow logic
@workflow(name="task_chain_workflow")
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2


@task(
    description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name only
""",
    # # Version with DaprChatClient (remember to uncomment the import at the top):
    # llm=DaprChatClient(component_name="openai")
)
def get_character() -> str:
    pass


@task(
    description="What is a famous line by {character}",
    # # Version with DaprChatClient:
    # llm=DaprChatClient(component_name="openai")
)
def get_line(character: str) -> str:
    pass


if __name__ == "__main__":
    wfapp = WorkflowApp()

    # Enhanced version with DaprChatClient for tool calling (remember to uncomment the import at the top):

    # # Configure DaprChatClient for specific provider
    # enhanced_client = DaprChatClient(component_name="openai")  # or "anthropic", "echo", etc.

    # wfapp = WorkflowApp(
    #     llm=enhanced_client
    # )

    results = wfapp.run_and_monitor_workflow_sync(task_chain_workflow)
    print(f"Famous Line: {results}")
