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

"""Shared helpers for the workflow event route tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import grpc
from pydantic import BaseModel

from dapr_agents.types.workflow import WorkflowEventRouteSpec

PATCH_TARGET = "dapr_agents.workflow.utils.registration.default_dapr_client_factory"


class FakeRpcError(grpc.RpcError):
    """A gRPC error with a fixed status code and details."""

    def __init__(self, code: Any, details: str = "") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> Any:
        return self._code

    def details(self) -> str:
        return self._details


class JobRef(BaseModel):
    workflow_id: str


class JobFinished(BaseModel):
    job: JobRef
    status: str


def workflow_state(status: Any) -> MagicMock:
    """A WorkflowState stand-in with the given runtime status."""
    state = MagicMock()
    state.runtime_status = status
    return state


def make_spec(defaults: dict[str, Any], **overrides: Any) -> WorkflowEventRouteSpec:
    """Build a spec on pub/sub ``messagepubsub`` / topic ``t`` from defaults + overrides."""
    values: dict[str, Any] = {"pubsub_name": "messagepubsub", "topic": "t"}
    values.update(defaults)
    values.update(overrides)
    return WorkflowEventRouteSpec(**values)
