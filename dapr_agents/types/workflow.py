from enum import Enum


class DaprWorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses for standardized tracking."""

    UNKNOWN = "unknown"  # Workflow is in an undefined state
    RUNNING = "running"  # Workflow is actively running
    COMPLETED = "completed"  # Workflow has completed
    FAILED = "failed"  # Workflow encountered an error
    TERMINATED = "terminated"  # Workflow was canceled or forcefully terminated
    SUSPENDED = "suspended"  # Workflow was temporarily paused
    PENDING = "pending"  # Workflow is waiting to start
