from dapr_agents import tool
from pydantic import BaseModel, Field
from typing import List


class GetWeatherSchema(BaseModel):
    location: str = Field(description="location to get weather for")


@tool(args_model=GetWeatherSchema)
def get_weather(location: str) -> str:
    """Get weather information based on location. It always returns 85F to be able to test the tool calling workflow."""

    return f"{location}: 85F."


class CalculateSchema(BaseModel):
    expression: str = Field(description="Arithmetic expression like '2+2' or '14*7+23'")


@tool(args_model=CalculateSchema)
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    import math
    import operator
    import re

    # Very basic evaluator supporting + - * / parentheses and integers
    tokens = re.findall(r"\d+|[()+\-*/]", expression.replace(" ", ""))
    if not tokens:
        return "Invalid expression"

    def precedence(op):
        return {"+": 1, "-": 1, "*": 2, "/": 2}.get(op, 0)

    def apply_op(a, b, op):
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        return str(ops[op](float(a), float(b)))

    values = []
    ops = []
    for tok in tokens:
        if tok.isdigit():
            values.append(tok)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                b = values.pop()
                a = values.pop()
                op = ops.pop()
                values.append(apply_op(a, b, op))
            ops.pop()
        else:
            while ops and precedence(ops[-1]) >= precedence(tok):
                b = values.pop()
                a = values.pop()
                op = ops.pop()
                values.append(apply_op(a, b, op))
            ops.append(tok)

    while ops:
        b = values.pop()
        a = values.pop()
        op = ops.pop()
        values.append(apply_op(a, b, op))

    return values[-1]


class SearchSchema(BaseModel):
    query: str = Field(description="web search query")
    limit: int = Field(default=3, ge=1, le=10, description="max results")


@tool(args_model=SearchSchema)
def web_search(query: str, limit: int = 3) -> List[str]:
    """Fake web search that returns example links for a query."""
    base = "https://example.org/search?q="
    return [f"{base}{query}&n={i+1}" for i in range(limit)]


tools = [get_weather, calculate, web_search]
