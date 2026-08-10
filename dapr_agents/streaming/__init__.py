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

"""Streaming transport layer for Dapr Agents.

Exposes the pluggable listener abstraction used by ``StreamEmitter`` to deliver
``AgentStreamChunk`` events across transports. Consumers pair with a listener
to drain chunks into application-specific surfaces (HTTP response, iterator,
webhook, etc.).
"""

from dapr_agents.streaming.consumers import (
    InProcessQueueConsumer,
    PubSubStreamConsumer,
    StreamConsumer,
)
from dapr_agents.streaming.emitter import StreamEmitter
from dapr_agents.streaming.keys import (
    INCLUDE_COMPLETE_MESSAGE,
    MESSAGE_METADATA,
    STREAM_CONTEXT,
    STREAM_LISTENER_CONFIG,
    STREAM_PHASE,
    StreamContextDict,
    USER_INPUT_EVENT_PREFIX,
)
from dapr_agents.streaming.listeners import (
    CompositeListener,
    InProcessQueueListener,
    PubSubListener,
    StreamListener,
    WebhookListener,
    build_listener,
    register_in_process_queue,
    register_stream_listener,
    unregister_in_process_queue,
)
from dapr_agents.types.streaming import AssistantMessageAccumulator

__all__ = [
    "AssistantMessageAccumulator",
    "CompositeListener",
    "INCLUDE_COMPLETE_MESSAGE",
    "InProcessQueueConsumer",
    "InProcessQueueListener",
    "MESSAGE_METADATA",
    "PubSubListener",
    "PubSubStreamConsumer",
    "STREAM_CONTEXT",
    "STREAM_LISTENER_CONFIG",
    "STREAM_PHASE",
    "StreamConsumer",
    "StreamContextDict",
    "StreamEmitter",
    "StreamListener",
    "USER_INPUT_EVENT_PREFIX",
    "WebhookListener",
    "build_listener",
    "register_in_process_queue",
    "register_stream_listener",
    "unregister_in_process_queue",
]
