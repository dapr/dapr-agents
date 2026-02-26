"""Integration tests for 09-agents-as-tools example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_multi_app


@pytest.mark.integration
class TestAgentsAsToolsExample:
    """Integration tests for 09-agents-as-tools example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key):
        self.example_dir = examples_dir / "09-agents-as-tools"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_cross_app(self, dapr_runtime):  # noqa: ARG002
        """Sam runs in a separate Dapr app; Frodo discovers and calls it as a tool."""
        dapr_yaml = self.example_dir / "dapr-cross-app.yaml"
        result = run_quickstart_or_examples_multi_app(
            dapr_yaml,
            cwd=self.example_dir,
            env={**self.env, "DAPR_HOST_IP": "127.0.0.1"},
            timeout=300,
            stream_logs=True,
        )

        assert result.returncode == 0, (
            f"Cross-app agents-as-tools failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert len(result.stdout) > 0 or len(result.stderr) > 0
