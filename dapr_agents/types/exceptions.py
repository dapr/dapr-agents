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
