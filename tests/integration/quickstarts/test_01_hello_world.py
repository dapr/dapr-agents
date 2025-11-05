"""Integration tests for 01-hello-world quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestHelloWorldQuickstart:
    """Integration tests for 01-hello-world quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "01-hello-world"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_basic_llm(self):
        """Test basic LLM example (01_ask_llm.py)."""
        script_path = self.quickstart_dir / "01_ask_llm.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert "Got response:" in result.stdout or "response" in result.stdout.lower()

    def test_simple_agent(self):
        """Test simple agent example (02_build_agent.py)."""
        script_path = self.quickstart_dir / "02_build_agent.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_agent(self, dapr_runtime):  # noqa: ARG002
        """Test durable agent example (03_durable_agent.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script_path = self.quickstart_dir / "03_durable_agent.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
            use_dapr=True,
            app_id="stateful-llm",
            dapr_http_port=3500,
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_chain_tasks(self, dapr_runtime):
        """Test chain tasks workflow (04_chain_tasks.py)."""
        script_path = self.quickstart_dir / "04_chain_tasks.py"
        result = run_quickstart_script(
            script_path,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="dapr-agent-wf",
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script_path}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
