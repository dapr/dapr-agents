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

"""Backward-compatibility tests for the deprecated dapr_agents.tool.mcp.schema module.

The schema helpers moved to the Dapr Python SDK in PR #612, but the
original import paths remain as deprecation shims for the v1.0.x line.
"""

import sys
import warnings

import pytest


@pytest.fixture
def fresh_schema_import():
    """Drop the deprecated module from sys.modules so the next import re-runs
    its module-level code (and re-fires the DeprecationWarning)."""
    sys.modules.pop("dapr_agents.tool.mcp.schema", None)
    yield
    sys.modules.pop("dapr_agents.tool.mcp.schema", None)


def test_top_level_import_still_works_and_warns(fresh_schema_import):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from dapr_agents.tool.mcp import create_pydantic_model_from_schema  # noqa: F401
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "create_pydantic_model_from_schema" in str(w.message)
        for w in caught
    )


def test_submodule_import_still_works_and_warns(fresh_schema_import):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from dapr_agents.tool.mcp.schema import create_pydantic_model_from_schema  # noqa: F401
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_type_mapping_re_exported():
    from dapr_agents.tool.mcp.schema import TYPE_MAPPING as shim_mapping
    from dapr.ext.workflow.mcp_schema import TYPE_MAPPING as sdk_mapping

    assert shim_mapping is sdk_mapping
    assert shim_mapping["integer"] is int
    assert shim_mapping["null"] is type(None)


def test_create_pydantic_model_from_schema_is_sdk_function():
    from dapr_agents.tool.mcp.schema import create_pydantic_model_from_schema as shim
    from dapr.ext.workflow.mcp_schema import (
        create_pydantic_model_from_schema as sdk_fn,
    )

    assert shim is sdk_fn

    model = shim(
        {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        "Compat",
    )
    assert model(x=5).x == 5
