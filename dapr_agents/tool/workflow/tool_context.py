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
        Pop workflow context and any hidden runtime kwargs out of kwargs, validate
        the remaining args against args_model, then re-attach the hidden kwargs so
        the executor receives them without exposing them in the LLM tool schema.

        Hidden kwargs stripped here:
          - ``ctx``              — Dapr workflow context (required)
          - ``_source_agent``    — name of the calling agent, used for "on-behalf-of"
                                    labelling (optional)
          - ``_child_instance_id`` — explicit instance ID for child workflows
                                      (optional, used by AgentWorkflowTool)
        """
        ctx = kwargs.pop(self.context_kwarg, None)
        if ctx is None:
            raise ToolError(
                f"Missing workflow context. Pass it as '{self.context_kwarg}=<DaprWorkflowContext>'."
            )
        source_agent = kwargs.pop("_source_agent", None)
        child_instance_id = kwargs.pop("_child_instance_id", None)

        validated = super()._validate_and_prepare_args(func, *args, **kwargs)
        validated[self.context_kwarg] = ctx
        if source_agent is not None:
            validated["_source_agent"] = source_agent
        if child_instance_id is not None:
            validated["_child_instance_id"] = child_instance_id
        return validated
