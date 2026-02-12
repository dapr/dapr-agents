import logging
from typing import Any, Callable, Dict, Optional

from dapr_agents.tool.base import AgentTool
from dapr_agents.types import ToolError

logger = logging.getLogger(__name__)


class WorkflowContextInjectedTool(AgentTool):
    """
    AgentTool variant that allows the *agent* to inject a Dapr workflow context
    into tool execution without exposing that context as part of the tool schema.

    The injected context is passed via a dedicated kwarg (default: "ctx").
    It is *not* validated by args_model and is omitted from args_schema.
    """

    # Name of the kwarg used to pass the workflow context at execution time.
    context_kwarg: str = "ctx"

    def _validate_and_prepare_args(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Pop workflow context out of kwargs, validate the remaining args against args_model,
        then re-attach the context so the executor receives it.
        """
        ctx = kwargs.pop(self.context_kwarg, None)
        if ctx is None:
            raise ToolError(
                f"Missing workflow context. Pass it as '{self.context_kwarg}=<DaprWorkflowContext>'."
            )

        validated = super()._validate_and_prepare_args(func, *args, **kwargs)
        validated[self.context_kwarg] = ctx
        return validated
