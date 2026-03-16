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

"""Integration tests for 02-durable-agent-tool-call example."""

import pytest
from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script


@pytest.mark.integration
class TestDurableAgentToolCallQuickstart:
    """Integration tests for 02-durable-agent-tool-call example."""

    @pytest.fixture(autouse=True)
    def setup(self, examples_dir, openai_api_key, is_ollama):
        """Setup test environment."""
        import os

        self.quickstart_dir = examples_dir / "02-durable-agent-tool-call"
        self.env = {"OPENAI_API_KEY": openai_api_key}
        if is_ollama:
            self.env["OPENAI_MODEL"] = os.environ["OLLAMA_MODEL"]
            self.env["OPENAI_BASE_URL"] = os.environ["OLLAMA_ENDPOINT"]

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
        result = run_quickstart_or_examples_script(
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
        result = run_quickstart_or_examples_script(
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
        result = run_quickstart_or_examples_script(
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
        result = run_quickstart_or_examples_script(
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
        result = run_quickstart_or_examples_script(
            script,
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="durableweatherapp",
            dapr_http_port=3500,
            app_port=8001,
            trigger_curl={
                "url": "http://localhost:8001/agent/run",
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
        result = run_quickstart_or_examples_script(
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
