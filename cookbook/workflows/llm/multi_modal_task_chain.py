from dapr_agents import OpenAIChatClient, NVIDIAChatClient
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.workflow import WorkflowApp, task, workflow
from dotenv import load_dotenv
import os
import logging

load_dotenv()

nvidia_llm = NVIDIAChatClient(
    model="meta/llama-3.1-8b-instruct", api_key=os.getenv("NVIDIA_API_KEY")
)

oai_llm = OpenAIChatClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    base_url=os.getenv("OPENAI_API_BASE_URL"),
)

azoai_llm = OpenAIChatClient(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="gpt-4o-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_api_version="2024-12-01-preview",
)


@workflow
def test_workflow(ctx: DaprWorkflowContext):
    """
    A simple workflow that uses a multi-modal task chain.
    """
    oai_results = yield ctx.call_activity(invoke_oai, input="Peru")
    azoai_results = yield ctx.call_activity(invoke_azoai, input=oai_results)
    nvidia_results = yield ctx.call_activity(invoke_nvidia, input=azoai_results)
    return nvidia_results


@task(
    description="What is the name of the capital of {country}?. Reply with just the name.",
    llm=oai_llm,
)
def invoke_oai(country: str) -> str:
    pass


@task(description="What is a famous thing about {capital}?", llm=azoai_llm)
def invoke_azoai(capital: str) -> str:
    pass


@task(
    description="Context: {context}. From the previous context. Pick one thing to do.",
    llm=nvidia_llm,
)
def invoke_nvidia(context: str) -> str:
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    wfapp = WorkflowApp()

    results = wfapp.run_and_monitor_workflow_sync(workflow=test_workflow)

    logging.info("Workflow results: %s", results)
    logging.info("Workflow completed successfully.")
