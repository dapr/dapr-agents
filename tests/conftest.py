"""Configuration for pytest."""
import pytest
import asyncio
import sys
import os
import tempfile
import shutil
from typing import Generator
from unittest.mock import MagicMock

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mock_module(name: str) -> MagicMock:
    """Create a mock module with the given name."""
    mock = MagicMock()
    mock.__name__ = name
    return mock


# Create mock modules for Dapr
mock_dapr = MagicMock()
mock_dapr.common = MagicMock()
mock_dapr.common.pubsub = MagicMock()
mock_dapr.common.pubsub.subscription = MagicMock()
mock_dapr.common.pubsub.subscription.StreamCancelledError = Exception
mock_dapr.common.pubsub.subscription.SubscriptionMessage = MagicMock


# Register the mock modules
sys.modules["dapr"] = mock_dapr
sys.modules["dapr.common"] = mock_dapr.common
sys.modules["dapr.common.pubsub"] = mock_dapr.common.pubsub
sys.modules["dapr.common.pubsub.subscription"] = mock_dapr.common.pubsub.subscription


class MockDaprPubSub:
    pass


class MockMessageRoutingMixin:
    pass


mock_messaging = MagicMock()
mock_messaging.__path__ = ["dapr_agents/workflow/messaging"]  # Make it a package
mock_messaging.__spec__ = MagicMock()
mock_messaging.DaprPubSub = MockDaprPubSub
mock_messaging.MessageRoutingMixin = MockMessageRoutingMixin

mock_routing = MagicMock()
mock_routing.MessageRoutingMixin = MockMessageRoutingMixin

# Register the messaging mocks
sys.modules["dapr_agents.workflow.messaging"] = mock_messaging
sys.modules["dapr_agents.workflow.messaging.routing"] = mock_routing


class MockWorkflowState:
    pass


class MockWorkflowRuntime:
    def activity(self, name):
        return lambda x: x


class MockDaprWorkflowClient:
    pass


class MockDaprWorkflowContext:
    """Mock DaprWorkflowContext for testing."""

    def __init__(self):
        self.instance_id = "test-instance"
        self.is_replaying = False
        self.call_activity = MagicMock()
        self.wait_for_external_event = MagicMock()
        self.create_timer = MagicMock()
        self.continue_as_new = MagicMock()


class MockWorkflowActivityContext:
    pass


class MockDaprClient:
    pass


class MockConversationInput:
    pass


class MockConversationResponse:
    pass


class MockStateItem:
    pass


class MockSubscription:
    pass


class StreamInactiveError(Exception):
    pass


mock_dapr.ext.workflow.WorkflowRuntime = MockWorkflowRuntime
mock_dapr.ext.workflow.DaprWorkflowClient = MockDaprWorkflowClient
mock_dapr.ext.workflow.DaprWorkflowContext = MockDaprWorkflowContext
mock_dapr.ext.workflow.WorkflowActivityContext = MockWorkflowActivityContext
mock_dapr.ext.workflow.workflow_state.WorkflowState = MockWorkflowState
mock_dapr.clients.DaprClient = MockDaprClient
mock_dapr.clients.grpc._request.ConversationInput = MockConversationInput
mock_dapr.clients.grpc._response.ConversationResponse = MockConversationResponse
mock_dapr.clients.grpc._state.StateItem = MockStateItem
mock_dapr.clients.grpc.subscription.StreamInactiveError = StreamInactiveError
mock_dapr.aio.clients.grpc.subscription.Subscription = MockSubscription

mock_modules = {
    "dapr": mock_dapr,
    "dapr.ext": mock_dapr.ext,
    "dapr.ext.fastapi": mock_dapr.ext.fastapi,
    "dapr.ext.workflow": mock_dapr.ext.workflow,
    "dapr.ext.workflow.workflow_state": mock_dapr.ext.workflow.workflow_state,
    "dapr.clients": mock_dapr.clients,
    "dapr.clients.grpc": mock_dapr.clients.grpc,
    "dapr.clients.grpc._request": mock_dapr.clients.grpc._request,
    "dapr.clients.grpc._response": mock_dapr.clients.grpc._response,
    "dapr.clients.grpc._state": mock_dapr.clients.grpc._state,
    "dapr.clients.grpc.subscription": mock_dapr.clients.grpc.subscription,
    "dapr.aio": mock_dapr.aio,
    "dapr.aio.clients": mock_dapr.aio.clients,
    "dapr.aio.clients.grpc": mock_dapr.aio.clients.grpc,
    "dapr.aio.clients.grpc.subscription": mock_dapr.aio.clients.grpc.subscription,
}

for name, mock in mock_modules.items():
    sys.modules[name] = mock

# This file is used by pytest to configure the test environment
# and provide shared fixtures across all tests.


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Mark all tests in this directory as asyncio tests
def pytest_collection_modifyitems(items):
    """Add asyncio marker to all test items."""
    for item in items:
        if (
            "test_" in item.name
            and hasattr(item.function, "__code__")
            and item.function.__code__.co_flags & 0x80
        ):
            item.add_marker(pytest.mark.asyncio)


@pytest.fixture(autouse=True)
def patch_openai_client(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", MagicMock())


@pytest.fixture(autouse=True)
def set_llm_component_default_env(monkeypatch):
    """Ensure DAPR_LLM_COMPONENT_DEFAULT is set for all tests."""
    monkeypatch.setenv("DAPR_LLM_COMPONENT_DEFAULT", "openai")


@pytest.fixture(autouse=True)
def set_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")


# Cleanup after all tests
def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests are done."""
    for name in mock_modules:
        if name in sys.modules:
            del sys.modules[name]
