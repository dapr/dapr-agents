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

"""A CloudEvent id reused for another target, through the real subscriber.

Each distinct target must receive its event exactly once, and a redelivery of
the identical message must still be deduplicated.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, Optional
from unittest.mock import patch

import grpc
import pytest

from dapr_agents.workflow.utils.registration import register_message_routes
from tests.workflow._event_route_helpers import PATCH_TARGET, FakeRpcError
from tests.workflow.test_workflow_event_routes_e2e_inprocess import (  # noqa: F401
    _event,
    _job,
    _spec,
)
from tests.workflow.test_workflow_event_routes_e2e_inprocess import env as _env

_WAIT = 5.0
_ID = "evt-shared"


@pytest.fixture
def env(monkeypatch):
    return _env.__wrapped__(monkeypatch)


class _Raises:
    """raise_workflow_event stand-in: call ``i`` hangs until ``release[i]``.

    ``outcome[i]`` (optional) is raised once the call is released. Every
    completed, successful call is recorded in ``landed``.
    """

    def __init__(self, hanging: int, outcome: Optional[dict[int, Exception]] = None):
        self.release = [threading.Event() for _ in range(hanging)]
        self.finished = [threading.Event() for _ in range(hanging)]
        self.outcome = outcome or {}
        self.attempts: list[str] = []
        self.landed: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, **kwargs: Any) -> None:
        with self._lock:
            index = len(self.attempts)
            self.attempts.append(kwargs["instance_id"])
        try:
            if index < len(self.release):
                self.release[index].wait(_WAIT)
                if index in self.outcome:
                    raise self.outcome[index]
            with self._lock:
                self.landed.append(kwargs["instance_id"])
        finally:
            if index < len(self.finished):
                self.finished[index].set()

    def land(self, index: int) -> None:
        self.release[index].set()
        assert self.finished[index].wait(_WAIT)


def _drive(env, raises: _Raises, script: Callable[[], Iterator[dict]]) -> Any:
    """Run the subscriber over ``script``'s messages; return the subscription mock."""
    mock_dapr, mock_wf = env
    mock_wf.raise_workflow_event.side_effect = raises
    done = threading.Event()

    def messages() -> Iterator[dict]:
        yield from script()
        done.set()

    sub = mock_dapr.subscribe.return_value
    sub.__iter__.side_effect = lambda: messages()
    with patch(PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr,
            routes=[_spec(call_timeout_seconds=0.05)],
            wf_client=mock_wf,
        )
    assert done.wait(_WAIT)
    for closer in closers:
        closer()
    return sub


def _msg(workflow_id: str) -> dict:
    return _event(_job(workflow_id=workflow_id), event_id=_ID)


def test_attack_a_reused_id_after_timed_out_raise_landed(env):
    raises = _Raises(hanging=1)

    def script() -> Iterator[dict]:
        yield _msg("wf-1")  # raise times out, lands in the background
        raises.land(0)
        yield _msg("wf-1")  # redelivery: acknowledged, not raised again
        yield _msg("wf-1")
        yield _msg("wf-2")  # same id, other target: must be raised
        yield _msg("wf-2")  # identical redelivery: deduplicated

    _drive(env, raises, script)
    assert sorted(raises.landed) == ["wf-1", "wf-2"]
    assert raises.attempts == ["wf-1", "wf-2"]


def test_attack_b_reused_id_does_not_evict_the_original(env):
    raises = _Raises(hanging=2)

    def script() -> Iterator[dict]:
        yield _msg("wf-1")  # times out
        raises.land(0)
        yield _msg("wf-2")  # same id, other target: also times out
        raises.land(1)
        yield _msg("wf-1")  # A's redelivery must not raise wf-1 again
        yield _msg("wf-1")
        yield _msg("wf-2")

    _drive(env, raises, script)
    assert sorted(raises.landed) == ["wf-1", "wf-2"]
    assert raises.attempts == ["wf-1", "wf-2"]


def test_attack_c_original_is_raised_after_a_reuse_succeeded(env):
    unavailable = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "down")
    raises = _Raises(hanging=1, outcome={0: unavailable})

    def script() -> Iterator[dict]:
        yield _msg("wf-1")  # times out; the raise then fails transiently
        raises.land(0)
        yield _msg("wf-2")  # same id, other target: succeeds
        yield _msg("wf-1")  # A's redelivery must still be raised; the second
        yield _msg("wf-1")  # covers the moment before A's outcome settles

    _drive(env, raises, script)
    assert sorted(raises.landed) == ["wf-1", "wf-2"]
    assert raises.attempts == ["wf-1", "wf-2", "wf-1"]
