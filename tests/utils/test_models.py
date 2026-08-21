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

"""Tests for model helper functions used by agents."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, Field, model_validator

from dapr_agents.utils.models import (
    get_model_factory,
    get_model_fields,
    is_pydantic_model,
    is_supported_model,
    is_supported_model_instance,
    merge_models,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class ExampleDataclass:
    """Dataclass fixture for helper tests."""

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


class ExampleModel(BaseModel):
    """Pydantic fixture for helper tests."""

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


# ---------------------------------------------------------------------------
# Model helper tests
# ---------------------------------------------------------------------------


class TestIsPydanticModel:
    """Tests for is_pydantic_model."""

    def test_is_pydantic_model(self):
        assert is_pydantic_model(ExampleModel)
        assert not is_pydantic_model(ExampleModel(name="base"))


class TestIsSupportedModel:
    """Tests for is_supported_model."""

    def test_is_supported_model(self):
        assert is_supported_model(dict)
        assert is_supported_model(ExampleDataclass)
        assert is_supported_model(ExampleModel)
        assert not is_supported_model(list)


class TestIsSupportedModelInstance:
    """Tests for is_supported_model_instance."""

    def test_is_supported_model_instance(self):
        assert is_supported_model_instance({})
        assert is_supported_model_instance(ExampleDataclass(name="base"))
        assert is_supported_model_instance(ExampleModel(name="base"))
        assert not is_supported_model_instance(ExampleModel)


class TestGetModelFields:
    """Tests for get_model_fields."""

    def test_get_model_fields_for_dict(self):
        fields = get_model_fields({"name": "base", "metadata": {}})
        assert list(fields) == ["name", "metadata"]

    def test_get_model_fields_for_dataclass(self):
        fields = get_model_fields(ExampleDataclass(name="base"))
        assert fields == ["name", "metadata", "description"]

    def test_get_model_fields_for_pydantic_model(self):
        fields = get_model_fields(ExampleModel(name="base"))
        assert list(fields) == ["name", "metadata", "description"]

    def test_get_model_fields_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported model type"):
            get_model_fields(object())


class TestGetModelFactory:
    """Tests for get_model_factory."""

    def test_get_model_factory_for_dict(self):
        factory = get_model_factory({"name": "base"})

        result = factory({"name": "updated", "metadata": {"k": "v"}})

        assert result == {"name": "updated", "metadata": {"k": "v"}}

    def test_get_model_factory_for_dataclass(self):
        factory = get_model_factory(ExampleDataclass(name="base"))

        result = factory({"name": "updated", "metadata": {"k": "v"}})

        assert result == ExampleDataclass(name="updated", metadata={"k": "v"})

    def test_get_model_factory_for_pydantic_model(self):
        factory = get_model_factory(ExampleModel(name="base"))

        result = factory({"name": "updated", "metadata": {"k": "v"}})

        assert result == ExampleModel(name="updated", metadata={"k": "v"})

    def test_get_model_factory_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported model type"):
            get_model_factory(object())


class TestMergeModels:
    """Tests for merge_models across supported model types."""

    def test_merge_models_dataclass_returns_merged(self):
        base = ExampleDataclass(
            name="base",
            metadata={"shared": "base", "base_only": "yes"},
            description="base-description",
        )
        override = ExampleDataclass(
            name="override",
            metadata={"shared": "override", "override_only": "yes"},
            description=None,
        )

        merged = merge_models(base, override)

        assert merged.name == "override"
        assert merged.metadata == {
            "shared": "override",
            "base_only": "yes",
            "override_only": "yes",
        }
        assert merged.description == "base-description"

    def test_merge_models_pydantic_returns_merged(self):
        base = ExampleModel(
            name="base",
            metadata={"shared": "base", "base_only": "yes"},
            description="base-description",
        )
        override = ExampleModel(
            name="override",
            metadata={"shared": "override", "override_only": "yes"},
            description=None,
        )

        merged = merge_models(base, override)

        assert merged.name == "override"
        assert merged.metadata == {
            "shared": "override",
            "base_only": "yes",
            "override_only": "yes",
        }
        assert merged.description == "base-description"

    def test_merge_models_returns_base_for_mismatched_types(self):
        base = ExampleDataclass(name="base")
        override = ExampleModel(name="override")

        merged = merge_models(base, override)

        assert merged is base

    def test_merge_models_returns_base_for_unsupported_override(self):
        base = ExampleDataclass(name="base")
        override = object()

        merged = merge_models(base, override)

        assert merged is base

    def test_merge_models_returns_override_for_unsupported_base(self):
        base = object()
        override = ExampleDataclass(name="override")

        merged = merge_models(base, override)

        assert merged is override

    def test_merge_models_dicts_shallow_merge(self):
        base = {
            "name": "base",
            "metadata": {"shared": "base", "base_only": "yes"},
        }
        override = {
            "metadata": {"override_only": "override"},
        }

        merged = merge_models(base, override)

        assert merged == {
            "name": "base",
            "metadata": {"override_only": "override"},
        }

    def test_merge_models_keeps_falsy_values_and_ignores_only_none(self):
        @dataclass
        class FalsyDataclass:
            enabled: bool
            count: int
            label: str
            tags: list[str]
            description: str | None = None

        base = FalsyDataclass(
            enabled=True,
            count=1,
            label="base",
            tags=["base"],
            description="base-description",
        )
        override = FalsyDataclass(
            enabled=False,
            count=0,
            label="",
            tags=[],
            description=None,
        )

        merged = merge_models(base, override)

        assert merged.enabled is False
        assert merged.count == 0
        assert merged.label == ""
        assert merged.tags == []
        assert merged.description == "base-description"

    def test_merge_models_failed_merge_raises(self):
        class DateRange(BaseModel):
            """Date range where start_date must be before end_date."""

            start_date: str | None = None
            end_date: str | None = None

            @model_validator(mode="after")
            def check_order(self) -> "DateRange":
                if (
                    self.start_date
                    and self.end_date
                    and self.start_date > self.end_date
                ):
                    raise ValueError("start_date must be before end_date")
                return self

        base = DateRange(start_date="2024-01-01", end_date="2024-06-01")
        override = DateRange(start_date="2024-12-01", end_date=None)

        result = merge_models(base, override)

        assert result == base
