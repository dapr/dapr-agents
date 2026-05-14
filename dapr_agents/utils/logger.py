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
import contextvars
from functools import wraps
from typing import Any, Callable

workflow_replaying_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "workflow_replaying_ctx", default=False
)


class WorkflowReplayFilter(logging.Filter):
    """
    A logging filter that suppresses log records if the Dapr workflow
    is currently replaying. This prevents duplicate logs during state rehydration.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not workflow_replaying_ctx.get()


def get_context_aware_logger(
    name: str, suppress_replay_logs: bool = True
) -> logging.Logger:
    """
    Returns a standard python logger configured with the WorkflowReplayFilter.
    Set suppress_replay_logs=False to retain standard workflow replay logging.
    """
    logger = logging.getLogger(name)
    has_filter = any(isinstance(f, WorkflowReplayFilter) for f in logger.filters)

    if suppress_replay_logs and not has_filter:
        logger.addFilter(WorkflowReplayFilter())
    elif not suppress_replay_logs and has_filter:
        logger.filters = [
            f for f in logger.filters if not isinstance(f, WorkflowReplayFilter)
        ]

    return logger


def with_logger_context(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to extract the DaprWorkflowContext from arguments
    and set the replay state in the context variable.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = None

        for arg in args:
            if hasattr(arg, "is_replaying"):
                ctx = arg
                break

        if ctx is None:
            for val in kwargs.values():
                if hasattr(val, "is_replaying"):
                    ctx = val
                    break

        if ctx is not None:
            token = workflow_replaying_ctx.set(ctx.is_replaying)
            try:
                return func(*args, **kwargs)
            finally:
                workflow_replaying_ctx.reset(token)

        return func(*args, **kwargs)

    return wrapper
