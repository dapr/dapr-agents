from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")


@mcp.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


@mcp.tool()
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b
