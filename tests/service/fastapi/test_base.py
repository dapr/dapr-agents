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

"""Unit tests for ``FastAPIServerBase.start``."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.service.fastapi.base import FastAPIServerBase

START_TIMEOUT = 5


class TestStart:
    @pytest.mark.asyncio
    async def test_server_failure_before_startup_is_raised(self):
        server = FastAPIServerBase(service_name="test-service")

        fake_uvicorn_server = MagicMock()
        fake_uvicorn_server.started = False

        async def failing_serve():
            raise OSError("address already in use")

        fake_uvicorn_server.serve = failing_serve

        with patch("uvicorn.Server", return_value=fake_uvicorn_server), pytest.raises(
            OSError, match="address already in use"
        ):
            await asyncio.wait_for(server.start(), START_TIMEOUT)
