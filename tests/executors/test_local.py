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

"""Unit tests for ``LocalCodeExecutor``."""

import asyncio
import sys
from pathlib import Path

import pytest

from dapr_agents.executors.local import LocalCodeExecutor

PROBE_TIMEOUT = 5


class TestGetMissingPackages:
    @pytest.mark.asyncio
    async def test_installed_package_is_not_reported_missing(self):
        executor = LocalCodeExecutor()
        env_path = Path(sys.executable).parent.parent
        missing = await asyncio.wait_for(
            executor._get_missing_packages(["sys"], env_path), PROBE_TIMEOUT
        )
        assert missing == []

    @pytest.mark.asyncio
    async def test_uninstalled_package_is_reported_missing(self):
        executor = LocalCodeExecutor()
        env_path = Path(sys.executable).parent.parent
        missing = await asyncio.wait_for(
            executor._get_missing_packages(
                ["definitely_not_a_real_package_xyz"], env_path
            ),
            PROBE_TIMEOUT,
        )
        assert missing == ["definitely_not_a_real_package_xyz"]
