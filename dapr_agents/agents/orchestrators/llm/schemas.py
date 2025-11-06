from functools import cached_property
import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from dapr_agents.agents.orchestrators.llm.state import PlanStep
from dapr_agents.llm.utils import StructureHandler


class IterablePlanStep(BaseModel):
    """
    A Pydantic model to capture IterablePlanStep objects.
    This wraps a list of PlanStep objects for structured output.
    """

    objects: List[PlanStep] = Field(description="A list of PlanStep objects")


class NextStep(BaseModel):
    """
    Represents the next step in a workflow, including the next agent to respond,
    an instruction message for that agent and the step id and substep id if applicable.
    """

    next_agent: str = Field(
        ..., description="The name of the agent selected to respond next."
    )
    instruction: str = Field(
        ..., description="A direct message instructing the agent on their next action."
    )
    step: int = Field(..., description="The step number the agent will be working on.")
    substep: Optional[float] = Field(
        None,
        description="The substep number (if applicable) the agent will be working on.",
    )


class TaskPlan(BaseModel):
    """Encapsulates the structured execution plan."""

    plan: List[PlanStep] = Field(..., description="Structured execution plan.")


class PlanStatusUpdate(BaseModel):
    step: int = Field(..., description="Step identifier (integer).")
    substep: Optional[float] = Field(
        None,
        description="Substep identifier (float, e.g., 1.1, 2.3). Set to None if updating a step.",
    )
    status: Literal["not_started", "in_progress", "blocked", "completed"] = Field(
        ..., description="Updated status for the step or sub-step."
    )


class ProgressCheckOutput(BaseModel):
    verdict: Literal["continue", "completed", "failed"] = Field(
        ...,
        description="Task status: 'continue' (in progress), 'completed' (done), or 'failed' (unresolved issue).",
    )
    plan_needs_update: bool = Field(
        ..., description="Indicates whether the plan requires updates (true/false)."
    )
    plan_status_update: Optional[List[PlanStatusUpdate]] = Field(
        None,
        description="List of status updates for steps or sub-steps. Each entry must contain `step`, optional `substep`, and `status`.",
    )
    plan_restructure: Optional[List[PlanStep]] = Field(
        None,
        description="A list of restructured steps. Only one step should be modified at a time.",
    )


# Schemas used in Prompts
class Schemas:
    """Lazily evaluated JSON schemas used in prompt calls."""

    @cached_property
    def plan(self) -> str:
        # Generate schema for IterablePlanStep (List[PlanStep] wrapper)
        iterable_plan_step = StructureHandler.create_iterable_model(PlanStep)
        return json.dumps(
            StructureHandler.enforce_strict_json_schema(
                iterable_plan_step.model_json_schema()
            )
        )

    @cached_property
    def progress_check(self) -> str:
        return json.dumps(
            StructureHandler.enforce_strict_json_schema(
                ProgressCheckOutput.model_json_schema()
            )
        )

    @cached_property
    def next_step(self) -> str:
        return json.dumps(NextStep.model_json_schema())


schemas = Schemas()
