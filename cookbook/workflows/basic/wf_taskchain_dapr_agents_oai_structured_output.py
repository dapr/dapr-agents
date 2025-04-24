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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    load_dotenv()

    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow_sync(workflow=question, input="Scooby Doo")

    print(results)
