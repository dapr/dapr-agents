"""Tests for substep normalization in LLM orchestrator schemas and plan utilities."""

import pytest

from dapr_agents.agents.orchestrators.llm.schemas import NextStep, PlanStatusUpdate
from dapr_agents.agents.orchestrators.llm.utils import find_step_in_plan


# ---------------------------------------------------------------------------
# PlanStatusUpdate validator
# ---------------------------------------------------------------------------


class TestPlanStatusUpdateSubstepValidator:
    """PlanStatusUpdate.normalize_substep should coerce whole-number floats to None."""

    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0, 3.0, 10.0])
    def test_whole_number_floats_become_none(self, value):
        update = PlanStatusUpdate(step=1, substep=value, status="completed")
        assert update.substep is None

    @pytest.mark.parametrize("value", [1.1, 1.2, 2.3, 3.14])
    def test_decimal_substeps_preserved(self, value):
        update = PlanStatusUpdate(step=1, substep=value, status="completed")
        assert update.substep == value

    def test_none_stays_none(self):
        update = PlanStatusUpdate(step=1, substep=None, status="completed")
        assert update.substep is None

    def test_omitted_substep_is_none(self):
        update = PlanStatusUpdate(step=1, status="completed")
        assert update.substep is None


# ---------------------------------------------------------------------------
# NextStep validator
# ---------------------------------------------------------------------------


class TestNextStepSubstepValidator:
    """NextStep.normalize_substep should coerce whole-number floats to None."""

    def test_whole_number_becomes_none(self):
        ns = NextStep(
            next_agent="agent-a",
            instruction="Do something",
            step=1,
            substep=0.0,
        )
        assert ns.substep is None

    def test_decimal_substep_preserved(self):
        ns = NextStep(
            next_agent="agent-a",
            instruction="Do something",
            step=1,
            substep=1.2,
        )
        assert ns.substep == 1.2


# ---------------------------------------------------------------------------
# find_step_in_plan with substep=None returns the parent step
# ---------------------------------------------------------------------------


class TestFindStepInPlan:
    """find_step_in_plan should return the parent step when substep is None."""

    @pytest.fixture()
    def plan(self):
        return [
            {
                "step": 1,
                "description": "First step",
                "status": "not_started",
                "substeps": [
                    {"substep": 1.1, "description": "Sub 1.1", "status": "not_started"},
                    {"substep": 1.2, "description": "Sub 1.2", "status": "not_started"},
                ],
            },
            {
                "step": 2,
                "description": "Second step",
                "status": "not_started",
            },
        ]

    def test_returns_parent_step_when_substep_none(self, plan):
        result = find_step_in_plan(plan, step=1, substep=None)
        assert result is not None
        assert result["step"] == 1
        assert result["description"] == "First step"

    def test_returns_substep_when_decimal(self, plan):
        result = find_step_in_plan(plan, step=1, substep=1.2)
        assert result is not None
        assert result["substep"] == 1.2

    def test_returns_none_for_missing_substep(self, plan):
        result = find_step_in_plan(plan, step=1, substep=0.0)
        assert result is None

    def test_returns_none_for_missing_step(self, plan):
        result = find_step_in_plan(plan, step=99, substep=None)
        assert result is None
