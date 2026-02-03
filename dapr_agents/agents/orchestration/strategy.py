"""Abstract base class for orchestration strategies.

This module defines the OrchestrationStrategy interface that all concrete
orchestration strategies must implement. The interface follows the Strategy
Pattern, allowing different orchestration algorithms to be plugged into
DurableAgent without modifying the core workflow logic.

Key Design Principles:
- Strategies are stateless (state is passed as parameters)
- Pure functions enable replay-safe workflows
- Each strategy manages its own state schema
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class OrchestrationStrategy(ABC):
    """Abstract base class for orchestration strategies.

    Orchestration strategies control how agents are selected and coordinated
    within multi-agent workflows. Each strategy implements a different
    approach to agent selection and task distribution.

    All methods receive state as a parameter and return updated state,
    making strategies stateless and replay-safe for Dapr Workflows.
    """

    @abstractmethod
    def initialize(self, ctx: Any, task: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize orchestration state for a new workflow.

        This method is called once at the start of orchestration to set up
        the initial state. For plan-based strategies, this might generate
        an execution plan. For simpler strategies, this might just prepare
        an agent list.

        Args:
            ctx: Workflow context (for logging, not for state storage)
            task: The task description to orchestrate
            agents: Dictionary of available agents {agent_name: metadata}

        Returns:
            Initial orchestration state dict. Schema is strategy-specific.

        Example state schemas:
            - Agent strategy: {"plan": [...], "task_history": [], "verdict": None}
            - RoundRobin: {"agent_names": [...], "last_response": None}
            - Random: {"agent_names": [...], "previous_agent": None}
        """
        pass

    @abstractmethod
    def select_next_agent(
        self, ctx: Any, state: Dict[str, Any], turn: int
    ) -> Dict[str, Any]:
        """Select the next agent to execute and prepare their instruction.

        This is the core decision-making method. Based on the current state
        and turn number, it determines which agent should act next and what
        instruction they should receive.

        Args:
            ctx: Workflow context (for logging, not for state storage)
            state: Current orchestration state from previous turn
            turn: Current iteration number (1-indexed)

        Returns:
            Action dict with keys:
                - "agent": str - Name of the next agent to execute
                - "instruction": str - Task/instruction for the agent
                - "metadata": dict - Strategy-specific metadata (e.g., step IDs)

        Raises:
            AgentError: If agent selection fails or no valid agent found
        """
        pass

    @abstractmethod
    def process_response(
        self, ctx: Any, state: Dict[str, Any], response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an agent's response and update orchestration state.

        After an agent completes their task, this method processes their
        response and updates the orchestration state accordingly. For
        plan-based strategies, this might update step statuses and check
        progress. For simpler strategies, this might just store the response.

        Args:
            ctx: Workflow context (for logging, not for state storage)
            state: Current orchestration state
            response: Agent's response message dict with "content", "name", etc.

        Returns:
            Result dict with keys:
                - "updated_state": dict - New orchestration state
                - "verdict": str - Optional continuation verdict
                    ("continue", "completed", "failed")

        Raises:
            AgentError: If response processing fails
        """
        pass

    @abstractmethod
    def should_continue(
        self, state: Dict[str, Any], turn: int, max_iterations: int
    ) -> bool:
        """Determine if orchestration should continue for another turn.

        This method checks if the workflow should proceed to the next iteration
        or stop. It considers the current state, turn count, and max iterations.

        Args:
            state: Current orchestration state
            turn: Current iteration number (1-indexed)
            max_iterations: Maximum allowed iterations

        Returns:
            True if orchestration should continue, False to stop

        Note:
            This method should check:
            - If max_iterations reached
            - If task completed (strategy-specific)
            - If task failed (strategy-specific)
        """
        pass

    @abstractmethod
    def finalize(
        self, ctx: Any, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final summary/output message for the caller.

        This method is called when orchestration completes (either successfully,
        failed, or max iterations reached). It generates a final message that
        will be returned to the caller (e.g., chat UI).

        Args:
            ctx: Workflow context (for logging, not for state storage)
            state: Final orchestration state

        Returns:
            Message dict with keys:
                - "role": "assistant"
                - "content": str - Final summary text
                - "name": str - Orchestrator agent name

        Note:
            For agent strategy, this typically uses LLM to generate a summary.
            For simpler strategies, this might return the last agent response
            or a concatenated summary.
        """
        pass
