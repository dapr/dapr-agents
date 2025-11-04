from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Type

from pydantic import BaseModel, Field

from dapr_agents.agents.schemas import TriggerAction
from dapr_agents.tool.base import AgentTool
from dapr_agents.types import AgentError
from dapr_agents.workflow.utils.pubsub import send_message_to_agent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HandoffSpec:
    """
    Declarative configuration for a durable agent handoff tool.

    Example:
        >>> HandoffSpec(
        ...     agent_name="Billing Specialist",
        ...     description="Delegate billing and payment issues",
        ...     default_task="Review the customer's billing inquiry"
        ... )
    """

    agent_name: str
    """Logical name of the downstream agent."""

    tool_name: Optional[str] = None
    """Optional tool name override (defaults to 'handoff_to_<target>')."""

    description: Optional[str] = None
    """Optional tool description override."""

    input_model: Optional[Type[BaseModel]] = None
    """Optional Pydantic model describing tool arguments."""

    default_task: Optional[str] = None
    """
    Fallback task instruction when the handoff tool is invoked without explicit task.

    Useful when LLMs forget to provide context:
        - Without: "Continue the conversation." (generic)
        - With: "Review the customer's billing inquiry" (specific)
    """


class HandoffToolInput(BaseModel):
    """
    Default payload model for durable handoff tools.

    Attributes:
        task: Instruction or follow-up to send to the target agent.
    """

    task: str = Field(
        description="Instruction or follow-up to provide the target agent.",
    )


def _function_style_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw.lower())
    parts = [part for part in cleaned.split("_") if part]
    if not parts:
        raise AgentError(
            "Target agent name must contain at least one alphanumeric character."
        )
    return "_".join(["handoff_to", *parts])


def _default_description(target_name: str) -> str:
    return f"Delegate control to '{target_name}'."


def create_handoff_tool(
    *,
    target_agent_name: str,
    source_agent_name: str,
    metadata_resolver: Callable[[], Mapping[str, Any]],
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    input_model: Optional[Type[BaseModel]] = None,
) -> AgentTool:
    """
    Build an AgentTool that publishes a TriggerAction to ``target_agent_name``.

    Args:
        target_agent_name: Logical name of the downstream agent to trigger.
        source_agent_name: Logical name of the emitting agent.
        metadata_resolver: Callable returning the latest registry metadata mapping. Should include
            topic/pubsub data for the target agent.
        tool_name: Optional explicit tool name (defaults to ``handoff_to_<target>``).
        description: Optional description override.
        input_model: Optional Pydantic model describing allowed tool arguments.

    Returns:
        AgentTool: A ready-to-register tool that performs the handoff.

    Raises:
        AgentError: If registry metadata is unavailable or the target agent is missing.
    """

    if not source_agent_name:
        raise AgentError("source_agent_name must be provided for handoff tools.")

    resolved_name = tool_name or _function_style_name(target_agent_name)
    resolved_description = description or _default_description(target_agent_name)
    resolved_model: Type[BaseModel] = input_model or HandoffToolInput

    async def _handoff_tool(task: str, **_: Any) -> Dict[str, Any]:
        trigger = TriggerAction(task=task)

        try:
            agents_metadata = metadata_resolver()
        except RuntimeError as exc:  # pragma: no cover - depends on configuration
            raise AgentError(str(exc)) from exc

        if target_agent_name not in agents_metadata:
            raise AgentError(
                f"Handoff target '{target_agent_name}' not found in registry for agent "
                f"'{source_agent_name}'."
            )

        await send_message_to_agent(
            source=source_agent_name,
            target_agent=target_agent_name,
            message=trigger,
            agents_metadata=agents_metadata,
        )

        logger.info(
            "Agent '%s' handed off to '%s'",
            source_agent_name,
            target_agent_name,
        )

        return {
            "handoff_to": target_agent_name,
            "task": task,
        }

    return AgentTool(
        name=resolved_name,
        description=resolved_description,
        args_model=resolved_model,
        func=_handoff_tool,
    )


def generate_handoff_aliases(tool_name: str, agent_name: str) -> set[str]:
    """
    Build a set of canonical aliases that should resolve to the same handoff target.

    Args:
        tool_name: Registered handoff tool name.
        agent_name: Logical name of the downstream agent.

    Returns:
        Set of alias strings (mixed case preserved) that can identify the handoff tool.
    """

    aliases: set[str] = {tool_name, tool_name.lower()}

    normalized_target = agent_name.strip()
    if normalized_target:
        aliases.add(normalized_target)
        aliases.add(normalized_target.lower())
        canonical_tool = _function_style_name(normalized_target)
        aliases.add(canonical_tool)
        aliases.add(canonical_tool.lower())

    return {alias for alias in aliases if alias}
