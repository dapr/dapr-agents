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
import pytest

from dapr_agents.utils.logger import (
    WorkflowReplayFilter,
    get_context_aware_logger,
    with_logger_context,
    workflow_replaying_ctx,
)


def test_workflow_replay_filter_logic():
    """Verify the filter suppresses records only when replaying context is True."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test Message",
        args=(),
        exc_info=None,
    )
    logger_filter = WorkflowReplayFilter()

    initial_token = workflow_replaying_ctx.set(False)

    try:
        assert logger_filter.filter(record) is True

        workflow_replaying_ctx.set(True)
        assert logger_filter.filter(record) is False
    finally:
        workflow_replaying_ctx.reset(initial_token)


def test_logger_initialization_idempotency():
    """Ensure get_context_aware_logger does not duplicate the filter on multiple calls."""
    logger_name = "idempotent_test_logger"
    logger = get_context_aware_logger(logger_name)

    def count_replay_filters(log_obj):
        return len([f for f in log_obj.filters if isinstance(f, WorkflowReplayFilter)])

    assert count_replay_filters(logger) == 1

    get_context_aware_logger(logger_name)
    assert count_replay_filters(logger) == 1


def test_with_logger_context_decorator_management():
    """Verify decorator extracts context and manages contextvar lifecycle."""

    class MockContext:
        def __init__(self, is_replaying: bool):
            self.is_replaying = is_replaying

    workflow_replaying_ctx.set(False)

    @with_logger_context
    def dummy_workflow(ctx, *args, **kwargs):
        return workflow_replaying_ctx.get()

    ctx_true = MockContext(is_replaying=True)
    assert dummy_workflow(ctx_true) is True
    assert workflow_replaying_ctx.get() is False

    ctx_false = MockContext(is_replaying=False)
    assert dummy_workflow(ctx_false) is False
    assert workflow_replaying_ctx.get() is False


def test_with_logger_context_error_handling():
    """Ensure context is reset even if the decorated function raises an error."""

    class MockContext:
        is_replaying = True

    @with_logger_context
    def failing_workflow(ctx):
        raise RuntimeError("Workflow failed")

    with pytest.raises(RuntimeError):
        failing_workflow(MockContext())

    assert workflow_replaying_ctx.get() is False


def test_logger_initialization_toggle():
    """Ensure the suppress_replay_logs flag correctly adds or removes the filter."""
    logger_name = "toggle_test_logger"

    logger = get_context_aware_logger(logger_name, suppress_replay_logs=False)
    assert not any(isinstance(f, WorkflowReplayFilter) for f in logger.filters)

    logger = get_context_aware_logger(logger_name, suppress_replay_logs=True)
    assert any(isinstance(f, WorkflowReplayFilter) for f in logger.filters)
