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

import functools
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.workflow.utils.core import sanitize_agent_name

logger = logging.getLogger(__name__)

AGENT_WORKFLOW_SUFFIX = "_agent_workflow"  # kept for backward compat


def agent_workflow_id(
    agent_name: str,
    framework: Optional[str] = None,
    workflow_name: Optional[str] = None,
) -> str:
    """
    Return the Dapr-registered workflow name for an agent.

    Priority order:
    1. If workflow_name is provided, use it directly
    2. If agent_name is already a full workflow name (starts with 'dapr.' and ends
       with '.workflow'), return it as-is
    3. If framework is provided and not "Dapr Agents", use 'dapr.{framework}.{agent_name}.workflow'
    4. Otherwise, use the standard format: 'dapr.agents.{agent_name}.workflow'

    Agent names are sanitized to comply with OpenAI's name requirements (no spaces,
    <, |, \\, /, >) before constructing workflow names.

    This supports:
    - Other agentic frameworks (e.g., "dapr.openai.catering-coordinator.workflow")
    - Dapr Agents framework (e.g., "dapr.agents.catering-coordinator.workflow")
    - Explicit workflow names passed directly

    Args:
        agent_name: The agent name (e.g., "catering-coordinator", "Samwise Gamgee")
        framework: Optional framework name (e.g., "openai", "pydantic_ai", "langgraph", "crewai").
            If None or "Dapr Agents", uses the standard dapr.agents.* format.
        workflow_name: Optional explicit workflow name. If provided, this takes precedence
            over all other parameters.

    Returns:
        The Dapr-registered workflow name with sanitized agent name.
    """
    from dapr_agents.tool.utils.function_calling import sanitize_openai_tool_name

    # Priority 1: Explicit workflow name
    if workflow_name:
        return workflow_name

    # Priority 2: Already a full workflow name
    if agent_name.startswith("dapr.") and agent_name.endswith(".workflow"):
        return agent_name

    # Sanitize agent name to comply with OpenAI requirements
    # This ensures workflow names don't contain invalid characters
    sanitized_agent_name = sanitize_openai_tool_name(agent_name)

    # Priority 3: Use framework if provided and not "Dapr Agents"
    if framework and framework != "Dapr Agents":
        # Normalize framework name (replace spaces/underscores with dots if needed)
        normalized_framework = framework.replace(" ", "-").replace("_", "-").lower()
        return f"dapr.{normalized_framework}.{sanitized_agent_name}.workflow"

    # Priority 4: Default to dapr.agents.* format
    return f"dapr.agents.{sanitized_agent_name}.workflow"


class AgentTaskArgs(BaseModel):
    """Arguments accepted by AgentWorkflowTool."""

    task: str = Field(
        ...,
        description="The instruction or task to send to the agent.",
    )


class AgentWorkflowTool(WorkflowContextInjectedTool):
    """
    A WorkflowContextInjectedTool that invokes another DurableAgent as a
    synchronous child workflow.

    The parent agent's LLM calls this tool with a ``task`` string; the tool
    schedules a child workflow named ``{target_agent_name}_agent_workflow``
    (optionally on a different Dapr app via ``target_app_id``) and waits for
    the child agent's final response.

    The workflow context (``ctx``) is injected by the dispatch loop and is
    never exposed in the LLM provider's function-call schema.
    """

    target_agent_name: str
    target_app_id: Optional[str] = None
    """
    Dapr app-id of the app hosting the target agent.
    ``None`` means the target agent is in the same Dapr app (in-process);
    cross-app invocation requires appropriate Dapr access-control policies.
    """


def _schedule_agent_workflow(
    ctx: Any,
    task: str,
    agent_name: str,
    agent_type: str = "agents",
    target_app_id: Optional[str] = None,
    _source_agent: Optional[str] = None,
    workflow_name: Optional[str] = None,
    framework: Optional[str] = None,
    _child_instance_id: Optional[str] = None,
) -> Any:
    """
    Schedule a child workflow for a named agent.

    This is intentionally a *sync* function that returns a Dapr workflow Task.
    The parent workflow yields on it::

        result = yield tool_obj(ctx=ctx, task="...")

    Args:
        ctx: Dapr workflow context supplied by the dispatch loop.
        task: The instruction to forward to the child agent.
        agent_name: Registered name of the target agent.
        agent_type: Framework/type prefix for the workflow name (e.g. "strands", "agents").
        target_app_id: Dapr app-id for cross-app routing; ``None`` for in-process.
        _source_agent: Name of the calling agent; forwarded in ``_message_metadata``
            so the child agent labels the user message as "on behalf of".
        workflow_name: Optional explicit workflow name. Takes precedence over framework.
        framework: Optional framework name for constructing workflow name.
        _child_instance_id: Explicit instance ID to assign to the child workflow.
            When provided, this exact ID is used so callers can record it in
            ``tool_history`` before the yield completes.
    """
    input_payload: dict = {"task": task}
    if _source_agent:
        input_payload["_message_metadata"] = {"source": _source_agent}

    workflow_id = agent_workflow_id(
        agent_name, framework=framework, workflow_name=workflow_name
    )

    call_kwargs: dict = {
        "workflow": workflow_id,
        "input": input_payload,
    }
    if target_app_id:
        call_kwargs["app_id"] = target_app_id
    if _child_instance_id:
        call_kwargs["instance_id"] = _child_instance_id

    logger.debug(
        "Scheduling child workflow '%s' app_id=%r task=%r",
        workflow_id,
        target_app_id,
        task,
    )
    return ctx.call_child_workflow(**call_kwargs)


def agent_to_tool(
    agent_name: str,
    description: str,
    *,
    agent_type: str = "agents",
    target_app_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    framework: Optional[str] = None,
) -> AgentWorkflowTool:
    """
    Create an AgentWorkflowTool for a named agent.

    This is the explicit factory for cases where you know the agent name and
    (optionally) its Dapr app ID — no registry lookup is performed.  Use it
    for cross-app agents where you have a known ``target_app_id``, or for
    advanced scenarios where you want full control.

    For registry-based auto-discovery, simply register both agents in the
    same registry; the parent agent's ``load_tools`` activity will pick up
    all registry peers automatically at workflow start.

    Args:
        agent_name: The name of the target agent (e.g., ``"catering-coordinator"``).
            This value is used to derive the tool name exposed to the LLM; the
            underlying :class:`AgentTool` normalizes it (title-casing, removing
            spaces and underscores), so e.g. ``"my agent"`` becomes ``"MyAgent"``.
            It should correspond to the agent's registered name under this
            normalization, rather than needing to match character-for-character.
        description: Human-readable description shown to the LLM in the tool
            schema (e.g. ``"Ring-bearer. Goal: carry the One Ring to Mordor."``).
        agent_type: Framework/type prefix for the workflow name (e.g. "strands",
            "agents").  Defaults to "agents" for backward compatibility.
        target_app_id: Dapr app-id of the app hosting the target agent.
            Pass ``None`` (default) for in-process invocation where both
            agents are registered in the same Dapr application.
        workflow_name: Optional explicit workflow name (e.g.,
            ``"dapr.openai.catering-coordinator.workflow"``). If provided, this
            takes precedence over framework-based construction.
        framework: Optional framework name (e.g., ``"openai"``, ``"pydantic_ai"``).
            Used to construct workflow name as ``dapr.{framework}.{agent_name}.workflow``.
            If None or "Dapr Agents", uses ``dapr.agents.{agent_name}.workflow``.

    Returns:
        AgentWorkflowTool ready to be registered in a DurableAgent's toolset.

    Example — explicit cross-app::

        from dapr_agents.tool.workflow import agent_to_tool

        sam_tool = agent_to_tool(
            "sam",
            "Logistics & Support. Goal: Manage provisions and supplies.",
            target_app_id="sam-app",
        )
        frodo = DurableAgent(name="frodo", tools=[sam_tool], ...)

    Example — with framework::

        catering_tool = agent_to_tool(
            "catering-coordinator",
            "Coordinates catering services.",
            framework="openai",
        )
        # This will call workflow: dapr.openai.CateringCoordinator.workflow
    """
    executor = functools.partial(
        _schedule_agent_workflow,
        agent_name=agent_name,
        agent_type=agent_type,
        target_app_id=target_app_id,
        workflow_name=workflow_name,
        framework=framework,
    )
    setattr(
        executor, "__name__", agent_name
    )  # partial has no __name__; AgentTool validator reads it
    return AgentWorkflowTool(
        name=agent_name,
        description=description,
        func=executor,
        args_model=AgentTaskArgs,
        target_agent_name=agent_name,
        target_app_id=target_app_id,
    )
