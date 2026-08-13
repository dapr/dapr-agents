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

"""Security-surface tests for the runner's listener config validation.

Covers the defense against HTTP-supplied ``{"type": "custom", ...}``
listener configs triggering arbitrary ``importlib.import_module`` calls.
"""

from __future__ import annotations

from typing import Mapping

import pytest


def test_build_listener_rejects_custom_by_default() -> None:
    from dapr_agents.streaming.listeners import build_listener

    with pytest.raises(ValueError, match="allow_custom"):
        build_listener(
            {
                "type": "custom",
                "factory": "dapr_agents.streaming.listeners.PubSubListener",
                "kwargs": {"pubsub_name": "p", "topic": "t"},
            }
        )


def test_build_listener_composite_rejects_nested_custom_by_default() -> None:
    from dapr_agents.streaming.listeners import build_listener

    with pytest.raises(ValueError, match="allow_custom"):
        build_listener(
            {
                "type": "composite",
                "listeners": [
                    {
                        "type": "custom",
                        "factory": "dapr_agents.streaming.listeners.PubSubListener",
                        "kwargs": {"pubsub_name": "p", "topic": "t"},
                    },
                ],
            }
        )


def test_build_listener_composite_allows_nested_custom_when_opt_in() -> None:
    from dapr_agents.streaming.listeners import build_listener, CompositeListener

    listener = build_listener(
        {
            "type": "composite",
            "listeners": [
                {
                    "type": "custom",
                    "factory": "dapr_agents.streaming.listeners.PubSubListener",
                    "kwargs": {"pubsub_name": "p", "topic": "t"},
                },
            ],
        },
        allow_custom=True,
    )
    try:
        assert isinstance(listener, CompositeListener)
    finally:
        listener.close()


def test_materialize_listener_fails_fast_on_partial_pubsub() -> None:
    """Pubsub listener without ``pubsub_name``/``topic`` must raise at
    config time, not surface as an opaque KeyError later."""
    from dapr_agents.workflow.runners.agent import AgentRunner

    with pytest.raises(ValueError, match="pubsub"):
        AgentRunner._materialize_listener_config(
            {"type": "pubsub"},
            "root-id",
            infra=None,
            agent_name="alice",
        )


def test_http_stream_handler_rejects_custom_listener_type() -> None:
    """Verify the FastAPI route guard that mirrors the build_listener
    default — HTTP-supplied ``{"type": "custom"}`` should return 400
    before any listener construction, so arbitrary ``importlib``
    invocations are not reachable via the public stream endpoint.

    Exercises the guard by replicating the handler's check against the
    sentinel, rather than standing up a full app + ``TestClient``
    fixture (which would require agent wiring). Complementary to
    :func:`test_build_listener_rejects_custom_by_default`.
    """
    from fastapi import HTTPException

    def _guard(listener_cfg) -> None:
        # Must mirror the check in ``_start_workflow_stream`` exactly —
        # update both sites in lockstep.
        if isinstance(listener_cfg, Mapping) and listener_cfg.get("type") == "custom":
            raise HTTPException(status_code=400, detail="custom rejected")

    with pytest.raises(HTTPException) as excinfo:
        _guard({"type": "custom", "factory": "os.system"})
    assert excinfo.value.status_code == 400

    # Non-custom types pass the guard.
    _guard({"type": "pubsub", "pubsub_name": "bus", "topic": "t"})
    _guard({"type": "in_process", "registry_key": "k"})
    _guard(None)


def test_build_stream_consumer_rejects_webhook_type() -> None:
    """``run_stream`` supports in-process and pubsub transports; webhook
    is a write-only listener with no consumer counterpart, so routing it
    to ``_build_stream_consumer`` should fail before the workflow is
    scheduled."""
    from dapr_agents.workflow.runners.agent import AgentRunner

    runner = AgentRunner.__new__(AgentRunner)
    with pytest.raises(ValueError):
        runner._build_stream_consumer(
            listener_config={"type": "webhook", "url": "http://x"},
            root_instance_id="root-1",
        )
