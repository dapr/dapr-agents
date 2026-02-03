import logging
import time
from typing import List

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents.llm.dapr import DaprChatClient

# Load environment variables (API keys, etc.)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Dapr workflow runtime and LLM client
runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


# ----- Models -----


class Question(BaseModel):
    """Represents a single research question."""

    text: str = Field(..., description="A research question related to the topic.")


class Questions(BaseModel):
    """Encapsulates a list of research questions."""

    questions: List[Question] = Field(
        ..., description="A list of research questions generated for the topic."
    )


# ----- Workflow -----


@runtime.workflow(name="research_workflow")
def research_workflow(ctx: DaprWorkflowContext, topic: str):
    """Defines a Dapr workflow for researching a given topic."""
    # 1) Generate research questions
    questions = yield ctx.call_activity(generate_questions, input={"topic": topic})

    # Extract question texts from the dictionary structure
    q_list = [q["text"] for q in questions["questions"]]

    # 2) Gather information for each question in parallel
    parallel_tasks = [
        ctx.call_activity(gather_information, input={"question": q}) for q in q_list
    ]
    research_results: List[str] = yield wf.when_all(parallel_tasks)

    # 3) Synthesize final report
    final_report: str = yield ctx.call_activity(
        synthesize_results, input={"topic": topic, "research_results": research_results}
    )

    return final_report


# ----- Activities -----


def generate_questions(ctx, topic: str) -> Questions:
    result = llm.generate(
        prompt=f"You are a research assistant. Generate exactly 3 focused research questions about the topic: {topic}. Return only a JSON object with a 'questions' list, each item having a 'text' field.",
        response_format=Questions,
    )
    try:
        questions = Questions.model_validate(result)
    except Exception as e:
        raise RuntimeError(f"LLM did not return valid Questions: {e}")
    return questions


@runtime.activity(name="gather_information")
def gather_information(ctx, question: str) -> str:
    return str(
        llm.generate(
            prompt=f"Research the following question and provide a detailed, well-cited answer (paragraphs + bullet points where helpful).\nQuestion: {question}\n"
        )
    )


@runtime.activity(name="synthesize_results")
def synthesize_results(ctx, topic: str, research_results: List[str]) -> str:
    return str(
        llm.generate(
            prompt=f"""
Create a comprehensive research report on the topic \"{topic}\" using the following research findings:\n\n{research_results}\n\nRequirements:\n- Clear executive summary (3-5 sentences)\n- Key findings (bulleted)\n- Risks/unknowns\n- Short conclusion\n\nReturn plain text (no JSON).\n"""
        )
    )


# ----- Entrypoint -----

if __name__ == "__main__":
    runtime.start()
    time.sleep(5)  # small grace period for runtime readiness

    client = wf.DaprWorkflowClient()
    research_topic = "The environmental impact of quantum computing"

    logging.info(f"Starting research workflow on: {research_topic}")
    instance_id = client.schedule_new_workflow(
        workflow=research_workflow,
        input=research_topic,
    )
    logging.info(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id, timeout_in_seconds=60)
    if not state:
        logging.error("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        logging.info(f"\nResearch Report:\n{state.serialized_output}")
    else:
        logging.error(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            logging.error("Failure type: %s", fd.error_type)
            logging.error("Failure message: %s", fd.message)
            logging.error("Stack trace:\n%s", fd.stack_trace)
        else:
            logging.error("Custom status: %s", state.serialized_custom_status)

    runtime.shutdown()
