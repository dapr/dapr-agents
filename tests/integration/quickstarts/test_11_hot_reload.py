"""Integration tests for 11_durable_agent_hot_reload quickstart."""

import os
import signal
import subprocess
import time

import pytest
from tests.integration.quickstarts.conftest import (
    _get_project_root,
    _setup_venv_and_python,
)


@pytest.mark.integration
class TestHotReloadQuickstart:
    """Integration tests for the durable agent hot-reload quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir
        self.env = {
            "OPENAI_API_KEY": openai_api_key,
        }

    def test_11_durable_agent_hot_reload(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent hot-reload example (11_durable_agent_hot_reload.py).

        The quickstart runs indefinitely. We start it, wait for the startup
        confirmation messages, then send SIGINT to gracefully shut it down.
        This verifies the agent initializes, starts, and subscribes to the
        configuration store without error.

        Note: dapr_runtime parameter ensures Dapr is initialized before this
        test runs.
        """
        script_path = self.quickstart_dir / "11_durable_agent_hot_reload.py"
        resources_path = self.quickstart_dir / "resources"
        full_env = os.environ.copy()
        full_env.update(self.env)
        project_root = _get_project_root(self.quickstart_dir)
        _, python_cmd = _setup_venv_and_python(
            self.quickstart_dir, project_root, create_venv=True
        )

        cmd = [
            "dapr",
            "run",
            "--app-id",
            "hot-reload-agent",
            "--dapr-http-port",
            "3500",
            "--resources-path",
            str(resources_path),
            "--",
            python_cmd,
            str(script_path.resolve()),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=self.quickstart_dir,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Give the agent time to start and log its initial state
            time.sleep(30)
        finally:
            # Gracefully stop
            proc.send_signal(signal.SIGINT)
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()

        combined = (stdout or "") + (stderr or "")
        assert len(combined) > 0
        # Verify the agent started with its initial role
        assert "Original Role" in combined
