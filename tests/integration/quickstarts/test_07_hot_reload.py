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

"""Integration tests for 07_durable_agent_hot_reload quickstart.

Tests verify that:
1. The agent starts with its initial configuration ("Original Role")
2. A live configuration change via Redis is picked up by the agent
3. The agent logs the updated role after hot-reload
4. Graceful shutdown works after config changes
"""

import logging
import os
import signal
import subprocess
import time

import pytest
from tests.integration.quickstarts.conftest import (
    _get_project_root,
    _setup_venv_and_python,
)

logger = logging.getLogger(__name__)


def _redis_cli_available() -> bool:
    """Check if redis-cli is available (standalone or via Docker)."""
    for cmd in (
        ["redis-cli", "ping"],
        ["docker", "exec", "dapr_redis", "redis-cli", "ping"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and "PONG" in result.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _redis_set(key: str, value: str) -> bool:
    """Set a key in Redis, trying standalone redis-cli first, then Docker."""
    for cmd_prefix in (
        ["redis-cli"],
        ["docker", "exec", "dapr_redis", "redis-cli"],
    ):
        try:
            result = subprocess.run(
                [*cmd_prefix, "SET", key, value],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _redis_del(key: str) -> None:
    """Delete a key from Redis (cleanup)."""
    for cmd_prefix in (
        ["redis-cli"],
        ["docker", "exec", "dapr_redis", "redis-cli"],
    ):
        try:
            subprocess.run(
                [*cmd_prefix, "DEL", key],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue


def _start_hot_reload_agent(
    quickstart_dir,
    resources_path,
    env: dict,
    python_cmd: str,
    app_id: str = "hot-reload-agent",
    dapr_http_port: int = 3501,
) -> subprocess.Popen:
    """Start the hot-reload agent quickstart with Dapr, returning the Popen handle."""
    script_path = quickstart_dir / "07_durable_agent_hot_reload.py"
    cmd = [
        "dapr",
        "run",
        "--app-id",
        app_id,
        "--dapr-http-port",
        str(dapr_http_port),
        "--resources-path",
        str(resources_path),
        "--",
        python_cmd,
        str(script_path.resolve()),
    ]
    full_env = os.environ.copy()
    full_env.update(env)

    return subprocess.Popen(
        cmd,
        cwd=quickstart_dir,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_and_collect(proc: subprocess.Popen, timeout: int = 30) -> str:
    """Send SIGINT and collect all output."""
    proc.send_signal(signal.SIGINT)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    return (stdout or "") + (stderr or "")


@pytest.mark.integration
class TestHotReloadQuickstart:
    """Integration tests for the durable agent hot-reload quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key, ollama_resources_dir, is_ollama):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir
        self.env = {
            "OPENAI_API_KEY": openai_api_key,
        }
        self.resources_path = ollama_resources_dir
        self.is_ollama = is_ollama
        if is_ollama:
            self.env["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

        project_root = _get_project_root(self.quickstart_dir)
        _, self.python_cmd = _setup_venv_and_python(
            self.quickstart_dir, project_root, create_venv=True
        )

    @pytest.mark.ollama
    def test_07_durable_agent_hot_reload_startup(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent hot-reload starts with initial configuration.

        Verifies the agent initializes, starts, and logs its initial role.
        """
        proc = _start_hot_reload_agent(
            self.quickstart_dir,
            self.resources_path,
            self.env,
            self.python_cmd,
            app_id="hot-reload-startup",
            dapr_http_port=3502,
        )

        try:
            time.sleep(30)
        finally:
            combined = _stop_and_collect(proc)

        assert len(combined) > 0
        assert "Original Role" in combined

    @pytest.mark.ollama
    def test_07_durable_agent_hot_reload_config_change(self, dapr_runtime):  # noqa: ARG002
        """Test that a live configuration change is picked up by the agent.

        This test:
        1. Starts the hot-reload agent
        2. Waits for it to initialize with "Original Role"
        3. Pushes a config change via Redis (agent_role -> "CI-Test-Role-Updated")
        4. Waits for the agent to log the updated role
        5. Verifies the change was applied

        Requires Redis to be available (started by `dapr init`).
        """
        if not _redis_cli_available():
            pytest.skip("redis-cli not available (needed for config change test)")

        new_role = "CI-Test-Role-Updated"
        # This matches RuntimeConfigKey.AGENT_ROLE from dapr_agents.agents.configs
        redis_key = "agent_role"

        # Clean up any stale config from previous runs
        _redis_del(redis_key)

        proc = _start_hot_reload_agent(
            self.quickstart_dir,
            self.resources_path,
            self.env,
            self.python_cmd,
            app_id="hot-reload-e2e",
            dapr_http_port=3503,
        )

        try:
            # Phase 1: Wait for agent to start and log initial role
            logger.info("Waiting for agent to start...")
            time.sleep(25)

            # Phase 2: Push a config change via Redis
            logger.info("Pushing config change: %s = %s", redis_key, new_role)
            success = _redis_set(redis_key, new_role)
            assert success, "Failed to SET config key in Redis"

            # Phase 3: Wait for the agent to process the config update
            # The quickstart logs "Current role: ..." every 5 seconds
            logger.info("Waiting for agent to pick up config change...")
            time.sleep(15)

        finally:
            combined = _stop_and_collect(proc)
            # Always clean up Redis key
            _redis_del(redis_key)

        assert len(combined) > 0, "No output from agent process"

        # Verify initial startup
        assert "Original Role" in combined, (
            f"Agent did not log initial role. Output:\n{combined[:2000]}"
        )

        # Verify config update was received and applied
        assert (
            f'applying config update: {redis_key}="{new_role}"' in combined
            or f"Current role: {new_role}" in combined
        ), (
            f"Agent did not apply config update to '{new_role}'.\n"
            f"Looked for 'applying config update: {redis_key}=\"{new_role}\"' "
            f"or 'Current role: {new_role}' in output.\n"
            f"Output (last 3000 chars):\n{combined[-3000:]}"
        )
