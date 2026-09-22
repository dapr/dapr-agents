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

import asyncio

import pytest

from dapr_agents.utils.signal_mixin import SignalHandlingMixin


class DummyService(SignalHandlingMixin):
    def __init__(self):
        super().__init__()
        self.stopped = False

    async def stop(self):
        self.stopped = True


@pytest.mark.asyncio
async def test_handle_shutdown_signal_schedules_async_graceful_shutdown():
    """A real SIGINT/SIGTERM callback must not silently drop the default
    async graceful_shutdown()/stop() cleanup path."""
    service = DummyService()
    service._loop = asyncio.get_running_loop()
    service._shutdown_event = asyncio.Event()

    service._handle_shutdown_signal(15)

    assert service._shutdown_event.is_set()
    # graceful_shutdown() is scheduled via call_soon_threadsafe, not awaited
    # inline, so give the loop a turn to run it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert service.stopped is True


def test_handle_shutdown_signal_without_captured_loop_does_not_raise():
    """Before setup_signal_handlers() runs (or if it never captured a loop),
    the signal callback must degrade gracefully instead of raising."""
    service = DummyService()
    service._shutdown_event = asyncio.Event()

    service._handle_shutdown_signal(2)

    assert service._shutdown_event.is_set()
    assert service.stopped is False


@pytest.mark.asyncio
async def test_setup_signal_handlers_captures_loop():
    class NoOpService(SignalHandlingMixin):
        pass

    service = NoOpService()
    assert service._loop is None

    service.setup_signal_handlers()

    assert service._loop is asyncio.get_running_loop()
    assert service._signal_handlers_setup is True


@pytest.mark.asyncio
async def test_handle_shutdown_signal_calls_sync_graceful_shutdown_directly():
    class SyncService(SignalHandlingMixin):
        def __init__(self):
            super().__init__()
            self.called = False

        def graceful_shutdown(self):
            self.called = True

    service = SyncService()
    service._shutdown_event = asyncio.Event()

    service._handle_shutdown_signal(15)

    assert service.called is True
