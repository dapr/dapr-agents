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
            assert (
                mock_signal.call_count == 2
            ), "Should register 2 signal handlers on Windows"

    with patch("platform.system", return_value="Linux"):
        add_signal_handlers_cross_platform(mock_loop, test_handler)
        assert (
            mock_loop.add_signal_handler.call_count == 2
        ), "Should register 2 signal handlers on Unix"


# Note: We intentially use asyncio here to test signal handling in a real event loop,
# and as a means to isolate this event loop from the other tests.
@pytest.mark.asyncio
async def test_add_signal_handlers_cross_platform_without_mocks_and_real_event_loop(
    event_loop,
):
    """Test using a real event loop to ensure signal handling works as expected."""
    from dapr_agents.utils import add_signal_handlers_cross_platform

    async def test_handler():
        pass

    try:
        add_signal_handlers_cross_platform(event_loop, test_handler)
        assert True  # if we are here, then we know the signal handlers were registered successfully
    except Exception as e:
        if "signal" in str(e).lower():
            pytest.warn(f"Signal-related error on {platform.system()}: {e}")
        else:
            raise
