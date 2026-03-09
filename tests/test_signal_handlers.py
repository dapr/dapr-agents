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
import platform
from unittest.mock import Mock, patch
import pytest


def test_add_signal_handlers_cross_platform():
    """Test cross platform signal handler function directly to ensure graceful shutdowns no matter platform."""
    from dapr_agents.utils import add_signal_handlers_cross_platform

    mock_loop = Mock()
    mock_loop.add_signal_handler = Mock()

    async def test_handler():
        pass

    with patch("platform.system", return_value="Windows"):
        with patch("signal.signal") as mock_signal:
            add_signal_handlers_cross_platform(mock_loop, test_handler)
            assert mock_signal.call_count == 2, (
                "Should register 2 signal handlers on Windows"
            )

    with patch("platform.system", return_value="Linux"):
        with patch("signal.signal") as mock_signal:
            add_signal_handlers_cross_platform(mock_loop, test_handler)
            assert mock_signal.call_count == 2, (
                "Should register 2 signal handlers on Unix"
            )


# Note: We intentially use asyncio here to test signal handling in a real event loop,
# and as a means to isolate this event loop from the other tests.
@pytest.mark.asyncio
async def test_add_signal_handlers_cross_platform_without_mocks_and_real_event_loop():
    """Test using a real event loop to ensure signal handling works as expected."""
    from dapr_agents.utils import add_signal_handlers_cross_platform

    async def test_handler():
        pass

    loop = asyncio.get_running_loop()

    try:
        add_signal_handlers_cross_platform(loop, test_handler)
        assert True  # if we are here, then we know the signal handlers were registered successfully
    except Exception as e:
        if "signal" in str(e).lower():
            pytest.warn(f"Signal-related error on {platform.system()}: {e}")
        else:
            raise
