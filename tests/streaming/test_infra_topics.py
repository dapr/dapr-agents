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

"""Tests for DaprInfra streaming topic helpers and config defaults."""

from __future__ import annotations

from dapr_agents.agents.components import DaprInfra
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
)
from dapr_agents.agents.schemas import AgentWorkflowEntry


class TestAgentExecutionConfigDefaults:
    def test_streaming_off_by_default(self) -> None:
        cfg = AgentExecutionConfig()
        assert cfg.streaming is False
        assert cfg.stream_listener is None

    def test_streaming_can_be_enabled(self) -> None:
        cfg = AgentExecutionConfig(streaming=True)
        assert cfg.streaming is True

    def test_stream_listener_accepts_dict(self) -> None:
        cfg = AgentExecutionConfig(
            streaming=True,
            stream_listener={"type": "pubsub", "pubsub_name": "bus", "topic": "t"},
        )
        assert cfg.stream_listener["type"] == "pubsub"


class TestAgentPubSubConfigDefaults:
    def test_stream_and_control_prefixes(self) -> None:
        ps = AgentPubSubConfig(pubsub_name="bus")
        assert ps.stream_topic_prefix == "agents.stream"
        assert ps.control_topic_prefix == "agents.control"

    def test_prefix_overrides(self) -> None:
        ps = AgentPubSubConfig(
            pubsub_name="bus",
            stream_topic_prefix="myapp.stream",
            control_topic_prefix="myapp.control",
        )
        assert ps.stream_topic_prefix == "myapp.stream"
        assert ps.control_topic_prefix == "myapp.control"


class TestDaprInfraStreamTopics:
    def test_with_pubsub(self) -> None:
        infra = DaprInfra(name="alice", pubsub=AgentPubSubConfig(pubsub_name="bus"))
        assert infra.stream_topic_name("abc-123") == "agents.stream.abc-123"
        assert infra.control_topic_name("abc-123") == "agents.control.abc-123"
        assert infra.stream_topic_prefix == "agents.stream"
        assert infra.control_topic_prefix == "agents.control"

    def test_without_pubsub(self) -> None:
        infra = DaprInfra(name="alice")
        assert infra.stream_topic_name("abc") is None
        assert infra.control_topic_name("abc") is None
        assert infra.stream_topic_prefix is None
        assert infra.control_topic_prefix is None

    def test_custom_prefixes_propagate(self) -> None:
        infra = DaprInfra(
            name="alice",
            pubsub=AgentPubSubConfig(
                pubsub_name="bus",
                stream_topic_prefix="tenant1.stream",
                control_topic_prefix="tenant1.control",
            ),
        )
        assert infra.stream_topic_name("xyz") == "tenant1.stream.xyz"
        assert infra.control_topic_name("xyz") == "tenant1.control.xyz"


class TestAgentWorkflowEntryStreamContext:
    def test_default_none(self) -> None:
        entry = AgentWorkflowEntry()
        assert entry.stream_context is None

    def test_accepts_nested_dict(self) -> None:
        entry = AgentWorkflowEntry(
            stream_context={
                "root_instance_id": "abc",
                "listener_config": {
                    "type": "pubsub",
                    "pubsub_name": "bus",
                    "topic": "agents.stream.abc",
                },
                "parent_agent": None,
                "depth": 0,
                "call_path": ["alice"],
            }
        )
        assert entry.stream_context["root_instance_id"] == "abc"
        assert entry.stream_context["listener_config"]["type"] == "pubsub"

    def test_model_dump_roundtrip(self) -> None:
        entry = AgentWorkflowEntry(stream_context={"root_instance_id": "xyz"})
        dumped = entry.model_dump(mode="json")
        assert dumped["stream_context"] == {"root_instance_id": "xyz"}
        restored = AgentWorkflowEntry.model_validate(dumped)
        assert restored.stream_context == {"root_instance_id": "xyz"}
