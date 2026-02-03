"""Random orchestration strategy with previous-speaker avoidance.

This strategy implements stochastic agent selection, randomly choosing the
next agent from available agents while avoiding consecutive selections of
the same agent (when possible).

Use this for:
- Exploring diverse agent perspectives
- Simulating unpredictable collaboration patterns
- Testing agent robustness to varied interaction sequences
"""

import logging
import random
from typing import Any, Dict, Optional

from dapr_agents.agents.orchestration.strategy import OrchestrationStrategy
from dapr_agents.types import AgentError

logger = logging.getLogger(__name__)


class RandomOrchestrationStrategy(OrchestrationStrategy):
    """Random orchestration strategy with avoidance logic.

    Agents are selected randomly from the available pool, with a preference
    to avoid selecting the same agent twice in a row (when multiple agents
    are available).

    State Schema:
        {
            "agent_names": List[str],        # List of agent names
            "previous_agent": Optional[str], # Last selected agent
            "last_response": Optional[Dict], # Last agent's response
            "task": str                      # Original task
        }
    """

    def initialize(self, ctx: Any, task: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize random orchestration state with agent list.

        Args:
            ctx: Workflow context (for logging)
            task: Task description
            agents: Available agents dict

        Returns:
            Initial state with agent names and no previous agent

        Raises:
            AgentError: If no agents available
        """
        if not agents:
            raise AgentError("No agents available for random orchestration")

        agent_names = list(agents.keys())

        if not ctx.is_replaying:
            logger.info(
                f"Initialized random orchestration with {len(agent_names)} agents: {agent_names}"
            )

        return {
            "agent_names": agent_names,
            "previous_agent": None,
            "last_response": None,
            "task": task,
        }

    def select_next_agent(
        self, ctx: Any, state: Dict[str, Any], turn: int
    ) -> Dict[str, Any]:
        """Select next agent randomly with previous-speaker avoidance.

        Args:
            ctx: Workflow context
            state: Current state with agent_names and previous_agent
            turn: Current turn number

        Returns:
            Action dict with randomly selected agent and instruction

        Raises:
            AgentError: If no agents in state
        """
        agent_names = state.get("agent_names", [])
        previous_agent = state.get("previous_agent")
        task = state.get("task", "")

        if not agent_names:
            raise AgentError("No agent names in random orchestration state")

        # Select agent randomly, avoiding previous agent if possible
        next_agent = self._select_random_agent(agent_names, previous_agent)

        # Build instruction based on whether there's a previous response
        last_response = state.get("last_response")
        if last_response:
            # Include context from previous agent
            previous_speaker = last_response.get("name", "previous agent")
            previous_content = last_response.get("content", "")
            instruction = (
                f"Task: {task}\n\n"
                f"Previous response from {previous_speaker}:\n{previous_content}\n\n"
                f"Continue working on this task based on the above context."
            )
        else:
            # First turn, just use the original task
            instruction = task

        if not ctx.is_replaying:
            logger.info(
                f"Random turn {turn}: Selected agent '{next_agent}' "
                f"(previous: {previous_agent or 'none'})"
            )

        return {
            "agent": next_agent,
            "instruction": instruction,
            "metadata": {
                "turn": turn,
                "previous_agent": previous_agent,
                "selection_method": "random",
            },
        }

    def _select_random_agent(
        self, agent_names: list, previous_agent: Optional[str]
    ) -> str:
        """Select a random agent, avoiding the previous agent if possible.

        Args:
            agent_names: List of available agent names
            previous_agent: Name of previously selected agent (or None)

        Returns:
            Randomly selected agent name
        """
        # If only one agent or no previous agent, just pick randomly
        if len(agent_names) == 1 or previous_agent is None:
            return random.choice(agent_names)

        # Try to avoid previous agent
        candidates = [name for name in agent_names if name != previous_agent]

        # If all agents filtered out (shouldn't happen), fall back to full list
        if not candidates:
            candidates = agent_names

        return random.choice(candidates)

    def process_response(
        self, ctx: Any, state: Dict[str, Any], response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store the agent's response and update previous_agent.

        Args:
            ctx: Workflow context
            state: Current state
            response: Agent's response message

        Returns:
            Updated state with last_response and previous_agent updated
        """
        agent_name = response.get("name", "unknown")

        if not ctx.is_replaying:
            logger.info(f"Processed response from agent '{agent_name}'")

        return {
            "updated_state": {
                **state,
                "last_response": response,
                "previous_agent": agent_name,
            },
            "verdict": "continue",  # Random always continues until max_iterations
        }

    def should_continue(
        self, state: Dict[str, Any], turn: int, max_iterations: int
    ) -> bool:
        """Check if orchestration should continue.

        Random orchestration continues until max_iterations is reached.

        Args:
            state: Current state
            turn: Current turn number
            max_iterations: Maximum allowed turns

        Returns:
            True if turn < max_iterations, False otherwise
        """
        should_continue = turn < max_iterations
        logger.debug(
            f"Random continuation check: turn={turn}, "
            f"max_iterations={max_iterations}, continue={should_continue}"
        )
        return should_continue

    def finalize(self, ctx: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final message with last agent's response.

        Args:
            ctx: Workflow context
            state: Final state with last_response

        Returns:
            Final message dict for caller
        """
        last_response = state.get("last_response")
        orchestrator_name = getattr(self, "orchestrator_name", "RandomOrchestrator")

        if last_response:
            # Return the last agent's response as the final result
            content = last_response.get("content", "")
            last_agent = last_response.get("name", "unknown")

            final_content = (
                f"Random orchestration completed.\n\n"
                f"Final response from {last_agent}:\n{content}"
            )
        else:
            # No responses collected (shouldn't happen in normal operation)
            final_content = "Random orchestration completed with no agent responses."

        if not ctx.is_replaying:
            logger.info("Random orchestration finalized")

        return {
            "role": "assistant",
            "content": final_content,
            "name": orchestrator_name,
        }
