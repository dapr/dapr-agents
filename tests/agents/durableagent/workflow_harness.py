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

"""
Drive a workflow through the pinned SDK's real orchestration executor.

Every step replays the whole recorded history from scratch and then applies
the new events, exactly as the sidecar does, so a replay mismatch surfaces as
a failed orchestration. Activities are answered by a test callback instead of
running; timers fire only when the test says so.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

# NOTE: these imports reach into private SDK modules (_durabletask, worker._Registry,
# _OrchestrationExecutor). They are tied to the pinned dapr-ext-workflow version and
# must be re-checked whenever that dependency is bumped.
from dapr.ext.workflow._durabletask import worker
from dapr.ext.workflow._durabletask.internal import helpers, protos as pb
from dapr.ext.workflow._durabletask.internal.timer import (
    is_optional_timer_action,
    new_timer_created_event,
    new_timer_fired_event,
)

ORCHESTRATOR_NAME = "under_test"
START_TIME = datetime(2024, 1, 1)

ActivityHandler = Callable[[str, Any], Any]


class WorkflowFailedError(Exception):
    """The orchestration failed, e.g. with a NonDeterminismError on replay."""


class ReplayHarness:
    """Records a history by running an orchestrator step by step."""

    def __init__(
        self,
        orchestrator: Callable[..., Any],
        on_activity: ActivityHandler,
        *,
        instance_id: str = "wf-replay",
        start_time: datetime = START_TIME,
        history: Optional[List[pb.HistoryEvent]] = None,
    ) -> None:
        self.registry = worker._Registry()
        self.registry.add_named_orchestrator(ORCHESTRATOR_NAME, orchestrator)
        self.on_activity = on_activity
        self.instance_id = instance_id
        self.now = start_time
        self.history: List[pb.HistoryEvent] = list(history or [])
        self.pending_timers: Dict[int, Any] = {}
        self.timers_created = 0
        self.done = False
        self.output: Any = None

    def start(self, workflow_input: Any = None) -> None:
        started = helpers.new_execution_started_event(
            ORCHESTRATOR_NAME, self.instance_id, json.dumps(workflow_input)
        )
        self._run([started])

    def resume(self) -> None:
        """Replay the recorded history with no new events beyond a restart."""
        self._run([])

    def raise_event(self, name: str, body: Any) -> None:
        if self.done:
            return  # the sidecar drops events raised on a completed workflow
        self._run([helpers.new_event_raised_event(name, json.dumps(body))])

    def fire_timers(self) -> None:
        fired = [
            new_timer_fired_event(tid, fire_at)
            for tid, fire_at in sorted(self.pending_timers.items())
        ]
        self.pending_timers = {}
        self.now += timedelta(minutes=1)
        self._run(fired)

    def _run(self, new_events: List[pb.HistoryEvent]) -> None:
        batch = [helpers.new_workflow_started_event(self.now)] + new_events
        while batch:
            executor = worker._OrchestrationExecutor(
                self.registry, logging.getLogger(__name__)
            )
            result = executor.execute(self.instance_id, self.history, batch)
            self.history.extend(batch)
            batch = []
            for action in result.actions:
                batch.extend(self._apply(action))
            if batch:
                batch.insert(0, helpers.new_workflow_started_event(self.now))

    def _apply(self, action: pb.WorkflowAction) -> List[pb.HistoryEvent]:
        if action.HasField("scheduleTask"):
            task = action.scheduleTask
            payload = json.loads(task.input.value) if task.input.value else None
            output = self.on_activity(task.name, payload)
            return [
                helpers.new_task_scheduled_event(
                    action.id, task.name, task.input.value
                ),
                helpers.new_task_completed_event(action.id, json.dumps(output)),
            ]
        if action.HasField("createTimer"):
            timer = action.createTimer
            origin = (
                getattr(timer, timer.WhichOneof("origin"))
                if timer.WhichOneof("origin")
                else None
            )
            if not is_optional_timer_action(action):
                self.pending_timers[action.id] = timer.fireAt
                self.timers_created += 1
            return [new_timer_created_event(action.id, timer.fireAt, origin)]
        if action.HasField("completeWorkflow"):
            done = action.completeWorkflow
            if done.workflowStatus == pb.ORCHESTRATION_STATUS_FAILED:
                raise WorkflowFailedError(done.failureDetails.errorMessage)
            self.done = True
            self.output = json.loads(done.result.value) if done.result.value else None
            return []
        raise AssertionError(f"unexpected action: {action}")
