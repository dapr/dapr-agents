"""Workflow-context injected tools.

This module defines a marker class for tools that must be invoked from within a
Dapr Workflow orchestrator.

Tools of this type typically accept a workflow context parameter named `ctx`
and return a workflow Task (e.g., the result of `ctx.call_child_workflow(...)`).

The durable agent orchestration loop can detect these tools and call them with
the current orchestrator context.
"""

from __future__ import annotations

from dapr_agents.tool.base import AgentTool


class WorkflowContextInjectedTool(AgentTool):
    """Marker subclass of :class:`~dapr_agents.tool.base.AgentTool`.

    Instances of this class are expected to be invoked from within a Dapr
    Workflow orchestrator (i.e., they require a `ctx` argument).
    """

    # Intentionally empty: used only for isinstance checks.
    pass
