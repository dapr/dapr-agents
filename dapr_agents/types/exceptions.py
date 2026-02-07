class AgentToolExecutorError(Exception):
    """Custom exception for AgentToolExecutor specific errors."""


class AgentError(Exception):
    """Custom exception for Agent specific errors, used to handle errors specific to agent operations."""


class ToolError(Exception):
    """Custom exception for tool-related errors."""


class StructureError(Exception):
    """Custom exception for errors related to structured handling."""


class FunCallBuilderError(Exception):
    """Custom exception for errors related to structured handling."""


class NotSupportedError(Exception):
    """Custom exception for errors related to not supported features or versions."""


class DaprRuntimeVersionNotSupportedError(NotSupportedError):
    """Custom exception for errors related to not supported Dapr runtime versions."""


class PubSubNotAvailableError(Exception):
    """Raised when a required PubSub component is not available in Dapr.

    This exception is raised during agent startup when the agent is configured
    to subscribe to a PubSub topic, but the PubSub component is not registered
    in the Dapr runtime.
    """

    def __init__(self, pubsub_name: str, topic: str, message: str | None = None):
        self.pubsub_name = pubsub_name
        self.topic = topic
        if message is None:
            message = (
                f"PubSub component '{pubsub_name}' is not available. "
                f"Cannot subscribe to topic '{topic}'. "
                "Ensure the PubSub component is configured and the Dapr sidecar is running."
            )
        super().__init__(message)
