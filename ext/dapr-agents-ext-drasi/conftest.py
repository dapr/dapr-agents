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

"""Fail loudly instead of silently skipping the Drasi extension tests.

The test modules under ``tests/`` skip themselves when ``dapr_agents.ext.drasi``
cannot be imported, which is right for a checkout without the ``drasi`` extra.
It also hid every test in CI: when these tests share a session with the
repository's ``tests/`` suite, ``tests/conftest.py`` puts the repository root on
``sys.path``, so ``dapr_agents`` resolves to the source tree and the installed
``dapr_agents.ext`` namespace is no longer found.

Set ``DAPR_AGENTS_REQUIRE_DRASI=1`` to turn that skip into a session failure.
This file sits outside ``tests/`` because both suites have a ``tests`` package,
and two ``tests.conftest`` modules cannot be registered in one session.
"""

import importlib
import os

import pytest

REQUIRE_DRASI_ENV_VAR = "DAPR_AGENTS_REQUIRE_DRASI"


def _drasi_import_error() -> ImportError | None:
    """Return the error from importing the extension, or None if it imports."""
    try:
        importlib.import_module("dapr_agents.ext.drasi")
    except ImportError as exc:
        return exc
    return None


if os.environ.get(REQUIRE_DRASI_ENV_VAR) == "1":
    error = _drasi_import_error()
    if error is not None:
        pytest.fail(
            f"{REQUIRE_DRASI_ENV_VAR}=1 but dapr_agents.ext.drasi cannot be "
            f"imported ({error!r}), so every Drasi extension test would be "
            "skipped. Install the extra with `uv sync --group test --extra "
            "drasi` and run these tests in their own session: "
            '`uv run pytest ext -m "not integration"`.',
            pytrace=False,
        )
