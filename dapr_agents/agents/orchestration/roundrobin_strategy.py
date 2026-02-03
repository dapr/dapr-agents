"""Round-robin orchestration strategy with deterministic agent selection.

This strategy implements simple sequential agent selection in a deterministic,
circular pattern. Each turn, the next agent in the sorted agent list is selected.

Use this for:
- Fair distribution of tasks across all agents
- Predictable, repeatable agent ordering
- Simple multi-agent collaboration without complex planning
"""

import logging
from typing import Any, Dict

from dapr_agents.agents.orchestration.strategy import OrchestrationStrategy
from dapr_agents.types import AgentError

logger = logging.getLogger(__name__)


class RoundRobinOrchestrationStrategy(OrchestrationStrategy):
    """Deterministic round-robin orchestration strategy.

    Agents are selected in a circular, sequential pattern based on sorted
    agent names. This ensures fair distribution and predictable ordering.

    State Schema:
        {
            "agent_names": List[str],        # Sorted list of agent names
            "last_response": Optional[Dict], # Last agent's response
            "task": str                      # Original task
        }
    """

    def initialize(self, ctx: Any, task: str, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize round-robin state with sorted agent list.

        Args:
            ctx: Workflow context (for logging)
            task: Task description
            agents: Available agents dict

        Returns:
            Initial state with sorted agent names

        Raises:
            AgentError: If no agents available
        """
        if not agents:
            raise AgentError("No agents available for round-robin orchestration")

        # Sort agent names for deterministic ordering
        agent_names = sorted(agents.keys())

        logger.info(
            f"Initialized round-robin orchestration with {len(agent_names)} agents: {agent_names}"
        )

        return {
            "agent_names": agent_names,
            "last_response": None,
            "task": task,
        }

    def select_next_agent(
        self, ctx: Any, state: Dict[str, Any], turn: int
    ) -> Dict[str, Any]:
        """Select next agent using modulo arithmetic for circular selection.

        Args:
            ctx: Workflow context
            state: Current state with agent_names
            turn: Current turn number (1-indexed)

        Returns:
            Action dict with selected agent and instruction

        Raises:
            AgentError: If no agents in state
        """
        agent_names = state.get("agent_names", [])
        task = state.get("task", "")

        if not agent_names:
            raise AgentError("No agent names in round-robin state")

        # Select agent using modulo for circular pattern
        # turn is 1-indexed, so we subtract 1 for 0-indexed array access
        agent_index = (turn - 1) % len(agent_names)
        next_agent = agent_names[agent_index]

        # Build instruction based on whether there's a previous response
        last_response = state.get("last_response")
        if last_response:
            # Include context from previous agent
            previous_agent = last_response.get("name", "previous agent")
            previous_content = last_response.get("content", "")
            instruction = (
                f"Task: {task}\n\n"
                f"Previous response from {previous_agent}:\n{previous_content}\n\n"
                f"Continue working on this task based on the above context."
            )
        else:
            # First turn, just use the original task
            instruction = task

        logger.info(
            f"Round-robin turn {turn}: Selected agent '{next_agent}' "
            f"(index {agent_index} of {len(agent_names)})"
        )

        return {
            "agent": next_agent,
            "instruction": instruction,
            "metadata": {
                "turn": turn,
                "agent_index": agent_index,
                "total_agents": len(agent_names),
            },
        }

    def process_response(
        self, ctx: Any, state: Dict[str, Any], response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store the agent's response for context in next turn.

        Args:
            ctx: Workflow context
            state: Current state
            response: Agent's response message

        Returns:
            Updated state with last_response stored
        """
        agent_name = response.get("name", "unknown")
        logger.info(f"Processed response from agent '{agent_name}'")

        return {
            "updated_state": {
                **state,
                "last_response": response,
            },
            "verdict": "continue",  # Round-robin always continues until max_iterations
        }

    def should_continue(
        self, state: Dict[str, Any], turn: int, max_iterations: int
    ) -> bool:
        """Check if orchestration should continue.

        Round-robin continues until max_iterations is reached.

        Args:
            state: Current state
            turn: Current turn number
            max_iterations: Maximum allowed turns

        Returns:
            True if turn < max_iterations, False otherwise
        """
        should_continue = turn < max_iterations
        logger.debug(
            f"Round-robin continuation check: turn={turn}, "
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
        orchestrator_name = getattr(self, "orchestrator_name", "RoundRobinOrchestrator")

        if last_response:
            # Return the last agent's response as the final result
            content = last_response.get("content", "")
            last_agent = last_response.get("name", "unknown")

            final_content = (
                f"Round-robin orchestration completed.\n\n"
                f"Final response from {last_agent}:\n{content}"
            )
        else:
            # No responses collected (shouldn't happen in normal operation)
            final_content = (
                "Round-robin orchestration completed with no agent responses."
            )

        logger.info("Round-robin orchestration finalized")

        return {
            "role": "assistant",
            "content": final_content,
            "name": orchestrator_name,
        }
