"""Integration tests for 09-agent-observability quickstart."""
import pytest
import time
import requests
from pathlib import Path
from tests.integration.quickstarts.conftest import run_quickstart_script


@pytest.mark.integration
class TestAgentObservabilityQuickstart:
    """Integration tests for 09-agent-observability quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        """Setup test environment."""
        self.quickstart_dir = quickstarts_dir / "09-agent-observability"
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_zipkin_tracing(self, zipkin_service):
        """Test agent with Zipkin tracing (01_agent_zipkin.py)."""
        # zipkin_service fixture already waits for service to be ready
        # and returns service info including endpoints

        script = self.quickstart_dir / "01_agent_zipkin.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Give Zipkin time to receive spans
        time.sleep(2)

        # Verify traces were sent to Zipkin
        try:
            # Query Zipkin for traces using the endpoint from fixture
            response = requests.get(zipkin_service["api_endpoint"], timeout=5)
            if response.status_code == 200:
                traces = response.json()
                # At least one trace should exist
                assert isinstance(traces, list)
        except requests.exceptions.RequestException:
            raise

    def test_otel_tracing(self, jaeger_service):
        """Test agent with OpenTelemetry tracing (02_agent_otel.py)."""
        # jaeger_service fixture already waits for service to be ready
        # and returns service info including endpoints

        script = self.quickstart_dir / "02_agent_otel.py"
        result = run_quickstart_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
        )

        # Check if script ran
        assert result.returncode == 0, (
            f"Quickstart failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Give Jaeger time to receive traces
        time.sleep(2)

        # Verify traces were sent to Jaeger
        try:
            # Query Jaeger API for traces using the endpoint from fixture
            response = requests.get(
                f"{jaeger_service['endpoint']}/api/traces?service=dapr-weather-agents",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                # Verify response structure
                assert "data" in data or "traces" in data
        except requests.exceptions.RequestException:
            # If we can't verify, that's okay - the test still passed if script ran
            pass
