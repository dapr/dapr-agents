"""
In-memory event store for demonstrating resumability functionality.

This is a robust, best-practice implementation intended for examples and testing,
not for production use where a persistent storage solution would be more appropriate.
"""

import logging
from collections import deque, defaultdict
from threading import Lock
from dataclasses import dataclass
from uuid import uuid4
from typing import Deque, Dict, Optional

from mcp.server.streamable_http import (
    EventCallback,
    EventId,
    EventMessage,
    EventStore,
    StreamId,
)
from mcp.types import JSONRPCMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventEntry:
    """
    Represents an event entry in the event store.
    """

    event_id: EventId
    stream_id: StreamId
    message: JSONRPCMessage


class InMemoryEventStore(EventStore):
    """
    Robust in-memory implementation of the EventStore interface for resumability.
    Keeps only the last N events per stream for memory efficiency.
    Thread-safe for concurrent access.
    Not for production use—use persistent storage for real deployments.
    """

    def __init__(self, max_events_per_stream: int = 100):
        """
        Args:
            max_events_per_stream: Maximum number of events to keep per stream.
        """
        self.max_events_per_stream: int = max_events_per_stream
        self.streams: Dict[StreamId, Deque[EventEntry]] = defaultdict(
            lambda: deque(maxlen=self.max_events_per_stream)
        )
        self.event_index: Dict[EventId, EventEntry] = {}
        self._lock = Lock()

    async def store_event(
        self, stream_id: StreamId, message: JSONRPCMessage
    ) -> EventId:
        """Store an event and return its event ID."""
        event_id = str(uuid4())
        event_entry = EventEntry(
            event_id=event_id, stream_id=stream_id, message=message
        )
        with self._lock:
            stream = self.streams[stream_id]
            if len(stream) == self.max_events_per_stream:
                oldest_event = stream[0]
                self.event_index.pop(oldest_event.event_id, None)
            stream.append(event_entry)
            self.event_index[event_id] = event_entry
        return event_id

    async def replay_events_after(
        self,
        last_event_id: EventId,
        send_callback: EventCallback,
    ) -> Optional[StreamId]:
        """Replay events that occurred after the specified event ID."""
        with self._lock:
            last_event = self.event_index.get(last_event_id)
            if not last_event:
                logger.warning(f"Event ID {last_event_id} not found in store")
                return None
            stream_id = last_event.stream_id
            stream_events = self.streams.get(stream_id, deque())
            found_last = False
            for event in stream_events:
                if found_last:
                    await send_callback(EventMessage(event.message, event.event_id))
                elif event.event_id == last_event_id:
                    found_last = True
            return stream_id
