import asyncio
import logging

from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext
from pydantic import BaseModel
from dotenv import load_dotenv


@workflow
def question(ctx: DaprWorkflowContext, input: int):
    step1 = yield ctx.call_activity(ask, input=input)
    return step1


class Dog(BaseModel):
    name: str
    bio: str
    breed: str


@task("Who was {name}?")
def ask(name: str) -> Dog:
    pass


async def main():
    logging.basicConfig(level=logging.INFO)

    # Load environment variables
    load_dotenv()

    # Initialize the WorkflowApp
    wfapp = WorkflowApp()

    # Run workflow
    result = await wfapp.run_and_monitor_workflow_async(
        workflow=question, input="Scooby Doo"
    )
    print(f"Results: {result}")


if __name__ == "__main__":
    asyncio.run(main())
