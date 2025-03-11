from dapr_agents.workflow import WorkflowApp, workflow, task
from dapr_agents.types import DaprWorkflowContext
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Define Workflow logic
@workflow(name='research_workflow')
def research_workflow(ctx: DaprWorkflowContext, topic: str):
    # Generate research questions
    questions = yield ctx.call_activity(generate_questions, input={"topic": topic})

    # Gather information for each question in parallel
    parallel_tasks = [
        ctx.call_activity(gather_information, input={"question": question})
        for question in questions
    ]
    research_results = yield ctx.when_all(parallel_tasks)

    # Synthesize the results into a coherent summary
    final_report = yield ctx.call_activity(synthesize_results,
                                           input={"topic": topic,
                                                  "research_results": research_results})

    return final_report

@task(description="Generate 3 focused research questions about {topic}. Return them as a list of strings.")
def generate_questions(topic: str) -> List[str]:
    pass

@task(description="Research information to answer this question: {question}. Provide a detailed response.")
def gather_information(question: str) -> str:
    pass

@task(description="Create a comprehensive research report on {topic} based on the following research: {research_results}")
def synthesize_results(topic: str, research_results: List[str]) -> str:
    pass

if __name__ == '__main__':
    wfapp = WorkflowApp()

    research_topic = "The environmental impact of quantum computing"

    print(f"Starting research workflow on: {research_topic}")
    results = wfapp.run_and_monitor_workflow(research_workflow, input=research_topic)
    print(f"\nResearch Report:\n{results}")
