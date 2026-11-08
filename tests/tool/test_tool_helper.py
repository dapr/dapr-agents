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

"""Tests for ToolHelper.infer_func_schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from dapr_agents import tool
from dapr_agents.tool.utils.tool import ToolHelper


@dataclass
class SearchFilter:
    tag: str


def sample_func(a: int, b: str = "x"):
    """A sample function with parameters."""
    return a


def future_annotated_func(
    tags: List[str],
    order: Literal["asc", "desc"] = "asc",
    search_filter: SearchFilter | None = None,
) -> str:
    """A function using PEP 563 postponed annotations and module-scoped types."""
    return f"{tags}:{order}:{search_filter}"


def test_infer_func_schema_uses_provided_name():
    model = ToolHelper.infer_func_schema(sample_func, name="MyCustomModel")
    assert model.__name__ == "MyCustomModel"


def test_infer_func_schema_default_name_not_last_param():
    model = ToolHelper.infer_func_schema(sample_func)
    assert model.__name__ == "sample_funcModel"


def test_infer_func_schema_fields_still_correct():
    model = ToolHelper.infer_func_schema(sample_func)
    assert set(model.model_fields) == {"a", "b"}


def test_infer_func_schema_resolves_postponed_annotations():
    model = ToolHelper.infer_func_schema(future_annotated_func)
    assert set(model.model_fields) == {"tags", "order", "search_filter"}
    instance = model(tags=["a", "b"], order="desc", search_filter={"tag": "release"})
    assert instance.tags == ["a", "b"]
    assert instance.order == "desc"
    assert instance.search_filter == SearchFilter(tag="release")

    wrapped = tool(future_annotated_func)
    schema = wrapped.to_function_call()
    props = schema["function"]["parameters"]["properties"]
    assert set(props) == {"tags", "order", "search_filter"}
