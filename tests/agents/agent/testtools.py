from dapr_agents import tool


@tool
def echo_tool(arg1: str) -> str:
    """Echoes the argument."""
    return arg1


@tool
def error_tool() -> str:
    """Tool that always raises an error."""
    raise RuntimeError("Tool failed")
