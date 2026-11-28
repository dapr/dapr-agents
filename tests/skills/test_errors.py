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

import pytest

from dapr_agents.skills import (
    SkillAccessError,
    SkillCatalogError,
    SkillError,
    SkillMalformedError,
    SkillNotFoundError,
    SkillSizeLimitError,
    SkillTransientAccessError,
    SkillUnsafePathError,
    SkillUnsupportedMediaTypeError,
)


@pytest.mark.parametrize(
    ("error_type", "category"),
    [
        (SkillNotFoundError, "not found"),
        (SkillMalformedError, "malformed"),
        (SkillUnsafePathError, "unsafe path"),
        (SkillAccessError, "access"),
        (SkillUnsupportedMediaTypeError, "unsupported media type"),
        (SkillSizeLimitError, "size limit"),
        (SkillTransientAccessError, "transient access"),
        (SkillCatalogError, "catalog"),
    ],
)
def test_public_skill_errors_have_stable_categories_and_context(
    error_type: type[SkillError], category: str
) -> None:
    error = error_type(
        skill_name="incident-investigation", path="references/checklist.md"
    )

    assert isinstance(error, SkillError)
    assert error.category == category
    assert "incident-investigation" in str(error)
    assert "references/checklist.md" in str(error)
    assert "secret-token" not in str(error)
