"""Integration tests for 03-standalone-agent-tool-call quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestStandaloneAgentToolCallQuickstart:
    """Integration tests for 03-standalone-agent-tool-call quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        import os

        self.quickstart_dir = quickstarts_dir / "03-standalone-agent-tool-call"
        self.env = {"OPENAI_API_KEY": openai_api_key}
        if os.getenv("HUGGINGFACE_API_KEY"):
            self.env["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
        if os.getenv("NVIDIA_API_KEY"):
            self.env["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

    def test_standalone_weather_agent_dapr(self, dapr_runtime):  # noqa: ARG002
        """Test standalone weather agent Dapr example (standalone_weather_agent_dapr.py)."""
        script = self.quickstart_dir / "standalone_weather_agent_dapr.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="standaloneweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_standalone_weather_agent_hf(self):
        """Test standalone weather agent HuggingFace example (standalone_weather_agent_hf.py)."""
        import os

        if not os.getenv("HUGGINGFACE_API_KEY"):
            pytest.skip("HUGGINGFACE_API_KEY not set")

        script = self.quickstart_dir / "standalone_weather_agent_hf.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_standalone_weather_agent_nv(self):
        """Test standalone weather agent NVIDIA example (standalone_weather_agent_nv.py)."""
        import os

        if not os.getenv("NVIDIA_API_KEY"):
            pytest.skip("NVIDIA_API_KEY not set")

        script = self.quickstart_dir / "standalone_weather_agent_nv.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_standalone_weather_agent_openai(self):
        """Test standalone weather agent OpenAI example (standalone_weather_agent_openai.py)."""
        script = self.quickstart_dir / "standalone_weather_agent_openai.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_standalone_weather_agent_tracing(self):
        """Test standalone weather agent tracing example (standalone_weather_agent_tracing.py)."""
        script = self.quickstart_dir / "standalone_weather_agent_tracing.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_standalone_agent_with_vectorstore(self):
        """Test standalone agent with vectorstore example (standalone_agent_with_vectorstore.py)."""
        script = self.quickstart_dir / "standalone_agent_with_vectorstore.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0
