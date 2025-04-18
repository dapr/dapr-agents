from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr_agents.types import DaprWorkflowContext
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

@workflow
def question(ctx:DaprWorkflowContext, input:int):
    step1 = yield ctx.call_activity(ask, input=input)
    return step1

class Dog(BaseModel):
    name: str
    bio: str
    breed: str

@task("Who was {name}?")
def ask(name:str) -> Dog:
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    load_dotenv()

    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow(workflow=question, input="Scooby Doo")
    print(results)