"""Integration tests for 03-durable-agent-tool-call quickstart."""
import pytest
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestDurableAgentToolCallQuickstart:
    """Integration tests for 03-durable-agent-tool-call quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        import os

        self.quickstart_dir = quickstarts_dir / "03-durable-agent-tool-call"
        self.env = {"OPENAI_API_KEY": openai_api_key}

        # Add optional API keys if they're set for local development
        if os.getenv("HUGGINGFACE_API_KEY"):
            self.env["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
        if os.getenv("NVIDIA_API_KEY"):
            self.env["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

    def test_durable_weather_agent_dapr(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent Dapr example (durable_weather_agent_dapr.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "durable_weather_agent_dapr.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Durable agents may take longer
            use_dapr=True,
            app_id="durableweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_hf(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent HuggingFace example (durable_weather_agent_hf.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        import os

        if not os.getenv("HUGGINGFACE_API_KEY"):
            pytest.skip("HUGGINGFACE_API_KEY not set")

        script = self.quickstart_dir / "durable_weather_agent_hf.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Durable agents may take longer
            use_dapr=True,
            app_id="durableweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_nv(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent NVIDIA example (durable_weather_agent_nv.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        import os

        if not os.getenv("NVIDIA_API_KEY"):
            pytest.skip("NVIDIA_API_KEY not set")

        script = self.quickstart_dir / "durable_weather_agent_nv.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Durable agents may take longer
            use_dapr=True,
            app_id="durableweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_openai(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent OpenAI example (durable_weather_agent_openai.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "durable_weather_agent_openai.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Durable agents may take longer
            use_dapr=True,
            app_id="durableweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_tracing(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent tracing example (durable_weather_agent_tracing.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "durable_weather_agent_tracing.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,  # Durable agents may take longer
            use_dapr=True,
            app_id="durableweatherapp",
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        # expect some output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_serve(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent serve example (durable_weather_agent_serve.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "durable_weather_agent_serve.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durableweatherapp",
            dapr_http_port=3500,
            app_port=8001,
            trigger_curl={
                "url": "http://localhost:8001/run",
                "method": "POST",
                "data": {"task": "What's the weather in New York?"},
                "headers": {"Content-Type": "application/json"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_durable_weather_agent_subscribe(self, dapr_runtime):  # noqa: ARG002
        """Test durable weather agent subscribe example (durable_weather_agent_subscribe.py).

        Note: dapr_runtime parameter ensures Dapr is initialized before this test runs.
        The fixture is needed for setup, even though we don't use the value directly.
        """
        script = self.quickstart_dir / "durable_weather_agent_subscribe.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durableweatherapp",
            dapr_http_port=3500,
            trigger_pubsub={
                "pubsub_name": "messagepubsub",
                "topic": "weather.requests",
                "data": {"task": "What's the weather in Boston?"},
                "wait_seconds": 5,
            },
        )

        assert result.returncode == 0, (
            f"Quickstart script '{script}' failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
