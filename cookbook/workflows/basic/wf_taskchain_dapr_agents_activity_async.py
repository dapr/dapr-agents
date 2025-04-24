import asyncio
import logging

from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr.ext.workflow import DaprWorkflowContext


@workflow(name="random_workflow")
def task_chain_workflow(ctx: DaprWorkflowContext, input: int):
    result1 = yield ctx.call_activity(step1, input=input)
    result2 = yield ctx.call_activity(step2, input=result1)
    result3 = yield ctx.call_activity(step3, input=result2)
    return [result1, result2, result3]


@task
def step1(activity_input: int) -> int:
    print(f"Step 1: Received input: {activity_input}.")
    return activity_input + 1


@task
def step2(activity_input: int) -> int:
    print(f"Step 2: Received input: {activity_input}.")
    return activity_input * 2


@task
def step3(activity_input: int) -> int:
    print(f"Step 3: Received input: {activity_input}.")
    return activity_input ^ 2


async def main():
    logging.basicConfig(level=logging.INFO)
    wfapp = WorkflowApp()

    result = await wfapp.run_and_monitor_workflow_async(task_chain_workflow, input=10)
    print(f"Results: {result}")


if __name__ == "__main__":
    asyncio.run(main())
