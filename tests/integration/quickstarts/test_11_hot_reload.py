"""Integration tests for 11_durable_agent_hot_reload quickstart."""

import pytest
from tests.integration.quickstarts.conftest import (
    run_quickstart_or_examples_script,
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
            "INTEGRATION_TEST": "1",
        }

    def test_11_durable_agent_hot_reload(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent hot-reload example (11_durable_agent_hot_reload.py).

        The script is started with INTEGRATION_TEST=1, which causes it to exit
        after printing the initial agent state. This verifies that the agent
        initializes, starts, and subscribes to the configuration store without
        error.

        Note: dapr_runtime parameter ensures Dapr is initialized before this
        test runs.
        """
        script_path = self.quickstart_dir / "11_durable_agent_hot_reload.py"
        result = run_quickstart_or_examples_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="hot-reload-agent",
            resources_path=self.quickstart_dir / "components",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert len(combined) > 0
        # Verify the agent started with its initial role
        assert "Original Role" in combined
