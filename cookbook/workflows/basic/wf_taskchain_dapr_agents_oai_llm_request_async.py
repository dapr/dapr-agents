import asyncio
import logging

from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv


# Define Workflow logic
@workflow(name="lotr_workflow")
def task_chain_workflow(ctx: DaprWorkflowContext):
    result1 = yield ctx.call_activity(get_character)
    result2 = yield ctx.call_activity(get_line, input={"character": result1})
    return result2


@task(
    description="""
    Pick a random character from The Lord of the Rings\n
    and respond with the character's name ONLY
"""
)
def get_character() -> str:
    pass


@task(
    description="What is a famous line by {character}",
)
def get_line(character: str) -> str:
    pass


async def main():
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Initialize the WorkflowApp
    wfapp = WorkflowApp()

    # Run workflow
    result = await wfapp.run_and_monitor_workflow_async(task_chain_workflow)
    print(f"Results: {result}")


if __name__ == "__main__":
    asyncio.run(main())
