"""Agent-based orchestration strategy using LLM planning and decision-making.

This strategy implements plan-based multi-agent orchestration where an LLM:
1. Generates a structured execution plan (Turn 1)
2. Selects the next agent and step based on plan progression
3. Validates step completion and updates plan status
4. Generates a final summary when orchestration completes

This is the most sophisticated orchestration mode, suitable for complex
tasks requiring adaptive planning and progress tracking.
"""

import json
import logging
from typing import Any, Dict, Optional

from dapr_agents.agents.orchestration.strategy import OrchestrationStrategy
from dapr_agents.agents.orchestrators.llm.prompts import (
    NEXT_STEP_PROMPT,
    PROGRESS_CHECK_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    TASK_PLANNING_PROMPT,
)
from dapr_agents.agents.orchestrators.llm.schemas import (
    IterablePlanStep,
    NextStep,
    ProgressCheckOutput,
    schemas,
)
from dapr_agents.agents.orchestrators.llm.utils import (
    find_step_in_plan,
    update_step_statuses,
)
from dapr_agents.common.exceptions import AgentError

logger = logging.getLogger(__name__)


class AgentOrchestrationStrategy(OrchestrationStrategy):
    """Plan-based orchestration using LLM for planning and decision-making.

    This strategy maintains an execution plan with steps and substeps,
    using an LLM to:
    - Generate the initial plan
    - Select the next agent/step at each turn
    - Validate and update plan progress
    - Generate final summaries

    State Schema:
        {
            "plan": List[Dict],              # List of PlanStep objects
            "task_history": List[Dict],      # Agent execution results
            "verdict": Optional[str]         # continue/completed/failed
        }
    """

    def __init__(self, llm: Any, activities: Any):
        """Initialize the agent orchestration strategy.

        Args:
            llm: LLM client for generating plans and decisions
            activities: Reference to DurableAgent activities (for call_llm, etc.)
        """
        self.llm = llm
        self.activities = activities

    def initialize(self, ctx: Any, task: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial execution plan using LLM.

        Args:
            ctx: Workflow context (for activities)
            task: Task description to plan for
            agents: Available agents dict

        Returns:
            Initial state with plan, empty task_history, and no verdict
        """
        # Call LLM to generate plan (using TASK_PLANNING_PROMPT)
        # This replicates lines 285-336 from durable.py
        plan_prompt = TASK_PLANNING_PROMPT.format(
            task=task, agents=agents, plan_schema=schemas.plan
        )

        # Note: In actual implementation, this will be called via ctx.call_activity
        # For now, we return the structure that the activity will populate
        return {
            "task": task,
            "agents": agents,
            "plan_prompt": plan_prompt,
            "response_format": IterablePlanStep.model_construct().model_dump_json(),
        }

    def select_next_agent(
        self, ctx: Any, state: Dict[str, Any], turn: int
    ) -> Dict[str, Any]:
        """Select next agent and step using LLM decision-making.

        Args:
            ctx: Workflow context
            state: Current state with plan and task_history
            turn: Current turn number

        Returns:
            Action dict with agent, instruction, step, and substep

        Raises:
            AgentError: If agent selection fails or plan is invalid
        """
        plan = state.get("plan", [])
        task = state.get("task", "")
        agents = state.get("agents", {})

        if not plan:
            raise AgentError("No plan available; cannot select next agent.")

        # Use NEXT_STEP_PROMPT to select next agent/step
        # This replicates lines 358-396 from durable.py
        next_step_prompt = NEXT_STEP_PROMPT.format(
            task=task,
            agents=agents,
            plan=plan,
            next_step_schema=schemas.next_step,
        )

        return {
            "task": task,
            "agents": agents,
            "plan": plan,
            "next_step_prompt": next_step_prompt,
            "response_format": NextStep.model_construct().model_dump_json(),
        }

    def process_response(
        self, ctx: Any, state: Dict[str, Any], response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process agent response and update plan status.

        Args:
            ctx: Workflow context
            state: Current state with plan
            response: Agent's response message

        Returns:
            Updated state with new plan status and verdict

        This replicates lines 431-487 from durable.py
        """
        plan = state.get("plan", [])
        task = state.get("task", "")
        step_id = state.get("current_step_id")
        substep_id = state.get("current_substep_id")

        # Mark step as completed
        target = find_step_in_plan(plan, step_id, substep_id)
        if target:
            target["status"] = "completed"
        plan = update_step_statuses(plan)

        # Use PROGRESS_CHECK_PROMPT to validate progress
        progress_prompt = PROGRESS_CHECK_PROMPT.format(
            task=task,
            plan=plan,
            step=step_id,
            substep=substep_id,
            results=response.get("content", ""),
            progress_check_schema=schemas.progress_check,
        )

        return {
            "task": task,
            "plan": plan,
            "step": step_id,
            "substep": substep_id,
            "result_content": response.get("content", ""),
            "progress_prompt": progress_prompt,
            "response_format": ProgressCheckOutput.model_construct().model_dump_json(),
        }

    def should_continue(
        self, state: Dict[str, Any], turn: int, max_iterations: int
    ) -> bool:
        """Check if orchestration should continue.

        Args:
            state: Current state with verdict
            turn: Current turn number
            max_iterations: Maximum allowed turns

        Returns:
            True if should continue, False to stop

        This replicates the logic from lines 488-511 in durable.py
        """
        verdict = state.get("verdict", "continue")

        # Stop if task completed or failed
        if verdict != "continue":
            return False

        # Stop if max iterations reached
        if turn >= max_iterations:
            return False

        return True

    def finalize(self, ctx: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final summary using LLM.

        Args:
            ctx: Workflow context
            state: Final state with plan and verdict

        Returns:
            Final message dict for caller

        This replicates lines 489-508 from durable.py
        """
        plan = state.get("plan", [])
        task = state.get("task", "")
        verdict = state.get("verdict", "continue")
        step_id = state.get("current_step_id")
        substep_id = state.get("current_substep_id")
        last_agent = state.get("last_agent", "")
        last_result = state.get("last_result", "")

        # If verdict is still "continue", we hit max iterations
        if verdict == "continue":
            verdict = "max_iterations_reached"

        # Use SUMMARY_GENERATION_PROMPT to create final summary
        summary_prompt = SUMMARY_GENERATION_PROMPT.format(
            task=task,
            verdict=verdict,
            plan=plan,
            step=step_id,
            substep=substep_id,
            agent=last_agent,
            result=last_result,
        )

        return {
            "task": task,
            "verdict": verdict,
            "plan": plan,
            "step": step_id,
            "substep": substep_id,
            "agent": last_agent,
            "result": last_result,
            "summary_prompt": summary_prompt,
        }
