"""Integration tests for 03-agent-tool-call quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestAgentToolCallQuickstart:
    """Integration tests for 03-agent-tool-call quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "03-agent-tool-call"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_weather_agent(self):
        """Test weather agent example (weather_agent.py)."""
        script = self.quickstart_dir / "weather_agent.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=90,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
