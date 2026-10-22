#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Durable deep agent: a research task you can crash and resume.

A lead agent writes a plan, hands each research question to a researcher
subagent, and saves its findings as notes in a Dapr state store through a
tool. The run is a Dapr workflow with a fixed instance id, so after a crash
the same command picks up where the run stopped.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import List

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dapr.ext.workflow import DaprWorkflowClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents import DurableAgent, tool
from dapr_agents.agents.configs import AgentExecutionConfig
from dapr_agents.hooks import Hooks, HookContext
from dapr_agents.workflow.runners import AgentRunner

NOTES_STORE = "research-notes"
RUN_ID = os.getenv("RESEARCH_RUN_ID", "deep-research-demo")
# Pause after each saved note so there is time to press Ctrl+C in the demo.
STEP_PAUSE_SECONDS = float(os.getenv("STEP_PAUSE_SECONDS", "8"))
DEFAULT_TOPIC = "the main trade-offs of event sourcing"

logger = logging.getLogger("deep-research")


def _key(suffix: str) -> str:
    return f"{RUN_ID}:{suffix}"


class WritePlanSchema(BaseModel):
    steps: List[str] = Field(description="Ordered research questions, one per step")


class SaveNoteSchema(BaseModel):
    step: int = Field(description="1-based number of the plan step this note answers")
    note: str = Field(description="The findings for that step, in a few sentences")


class ReadNoteSchema(BaseModel):
    step: int = Field(description="1-based number of the plan step to read")


@tool(args_model=WritePlanSchema)
def write_plan(steps: List[str]) -> str:
    """Save the research plan. Call this first, once."""
    with DaprClient() as client:
        client.save_state(NOTES_STORE, _key("plan"), json.dumps(steps))
    logger.info("TOOL write_plan: saved %d steps to %s", len(steps), NOTES_STORE)
    return f"Plan saved with {len(steps)} steps."


@tool(args_model=SaveNoteSchema)
def save_note(step: int, note: str) -> str:
    """Save the findings for one plan step to the notes store."""
    with DaprClient() as client:
        client.save_state(NOTES_STORE, _key(f"note:{step}"), note)
    logger.info("TOOL save_note: step %d saved to %s", step, NOTES_STORE)
    time.sleep(STEP_PAUSE_SECONDS)
    return f"Note for step {step} saved."


@tool
def list_notes() -> str:
    """List the plan steps and whether a note has been saved for each."""
    with DaprClient() as client:
        plan = json.loads(client.get_state(NOTES_STORE, _key("plan")).data or "[]")
        lines = []
        for number, question in enumerate(plan, start=1):
            saved = client.get_state(NOTES_STORE, _key(f"note:{number}")).data
            lines.append(f"{number}. [{'done' if saved else 'todo'}] {question}")
    logger.info("TOOL list_notes: %d plan steps", len(lines))
    return "\n".join(lines)


@tool(args_model=ReadNoteSchema)
def read_note(step: int) -> str:
    """Read the saved note for one plan step."""
    with DaprClient() as client:
        note = client.get_state(NOTES_STORE, _key(f"note:{step}")).data
    logger.info("TOOL read_note: step %d", step)
    return note.decode() if note else "No note saved for this step."


def llm_call_logger(agent_name: str) -> Hooks:
    """Log every LLM call. Replayed workflow history never reaches this hook."""
    calls = 0

    def log_llm_call(ctx: HookContext) -> None:
        nonlocal calls
        calls += 1
        messages = ctx.payload.get("messages", [])
        logger.info(
            "LLM CALL #%d agent=%s messages_in_context=%d",
            calls,
            agent_name,
            len(messages),
        )

    return Hooks(before_llm_call=[log_llm_call])


def build_agents() -> DurableAgent:
    # One workflow runtime shared by both agents, so the researcher runs as a
    # child workflow of the lead inside this process.
    runtime = wf.WorkflowRuntime()

    researcher = DurableAgent(
        name="researcher",
        role="Researcher. Answers one focused research question.",
        goal="Answer the question you are given accurately in under 120 words.",
        instructions=["Answer only the question asked. Do not ask follow-ups."],
        hooks=llm_call_logger("researcher"),
        runtime=runtime,
    )
    researcher.register(runtime)

    return DurableAgent(
        name="lead",
        role="Research Lead",
        goal="Produce a short research brief by working through a written plan.",
        instructions=[
            "First call write_plan with exactly 3 research questions.",
            "Then, for each plan step in order: call the researcher tool with that "
            "one question, then call save_note with its answer. Do one step at a "
            "time and do not repeat earlier steps.",
            "Do not copy earlier findings into later messages; they live in the notes.",
            "When all steps are saved, call list_notes, then read_note for each "
            "step, and write the final brief from those notes.",
        ],
        tools=[write_plan, save_note, list_notes, read_note, researcher],
        # Three steps need about 15 LLM turns; the default cap is 10.
        execution=AgentExecutionConfig(max_iterations=25),
        hooks=llm_call_logger("lead"),
        runtime=runtime,
    )


def existing_run_state(client: DaprWorkflowClient):
    try:
        return client.get_workflow_state(RUN_ID, fetch_payloads=False)
    except Exception:  # noqa: BLE001 - an unknown id surfaces differently per sidecar version
        return None


async def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    for noisy in ("durabletask-client", "durabletask-worker", "WorkflowRuntime"):
        logging.getLogger(noisy).propagate = False

    topic = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
    lead = build_agents()
    runner = AgentRunner()
    client = DaprWorkflowClient()

    try:
        # Start the workflow worker first: it resumes any unfinished run by itself.
        lead.start()
        previous = existing_run_state(client)
        if previous is not None:
            logger.info(
                "RESUME run %s found in workflow state (status=%s); not rescheduling",
                RUN_ID,
                previous.runtime_status.name,
            )
        else:
            logger.info("START run %s on topic: %s", RUN_ID, topic)
            await runner.run(
                lead,
                payload={"task": f"Write a short research brief on {topic}."},
                instance_id=RUN_ID,
                wait=False,
            )

        final = await asyncio.to_thread(
            client.wait_for_workflow_completion, RUN_ID, timeout_in_seconds=900
        )
        print(f"\nFinal brief:\n{final.serialized_output}\n", flush=True)
    finally:
        runner.shutdown(lead)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
