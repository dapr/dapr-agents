from time import sleep
import dapr.ext.workflow as wf

wfr = wf.WorkflowRuntime()


@wfr.workflow(name="random_workflow")
def task_chain_workflow(ctx: wf.DaprWorkflowContext, x: int):
    result1 = yield ctx.call_activity(step1, input=x)
    result2 = yield ctx.call_activity(step2, input=result1)
    result3 = yield ctx.call_activity(step3, input=result2)
    return [result1, result2, result3]


@wfr.activity
def step1(ctx, activity_input):
    print(f"Step 1: Received input: {activity_input}.")
    # Do some work
    return activity_input + 1


@wfr.activity
def step2(ctx, activity_input):
    print(f"Step 2: Received input: {activity_input}.")
    # Do some work
    return activity_input * 2


@wfr.activity
def step3(ctx, activity_input):
    print(f"Step 3: Received input: {activity_input}.")
    # Do some work
    return activity_input ^ 2


if __name__ == "__main__":
    wfr.start()
    sleep(5)  # wait for workflow runtime to start

    wf_client = wf.DaprWorkflowClient()
    instance_id = wf_client.schedule_new_workflow(
        workflow=task_chain_workflow, input=10
    )
    print(f"Workflow started. Instance ID: {instance_id}")
    state = wf_client.wait_for_workflow_completion(instance_id)
    print(f"Workflow completed! Status: {state.runtime_status}")

    wfr.shutdown()
