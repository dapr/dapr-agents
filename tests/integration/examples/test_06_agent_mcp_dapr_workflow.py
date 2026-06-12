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

"""Integration tests for the MCP + DaprWorkflow quickstart.

Requires:
  - dapr init completed (ensured by the dapr_runtime fixture)
  - Redis running locally (used by the Dapr components in resources/)
  - The dapr binary built from the MCPServer-enabled branch

Run:
    pytest tests/integration/examples/test_06_agent_mcp_dapr_workflow.py -v -m integration
"""

import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from tests.integration.quickstarts.conftest import run_quickstart_or_examples_script

WEATHER_SERVER_PORT = 8081
WEATHER_SERVER_PORT_2 = 8082


def _wait_for_port(port: int, timeout: float = 10.0) -> bool:
    """Return True once something is listening on *port*, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def weather_mcp_server(quickstarts_dir):
    """Start the weather MCP server subprocess and wait until it is ready."""
    proc = subprocess.Popen(
        [
            sys.executable,
            str(quickstarts_dir / "weather_mcp_server.py"),
            "--port",
            str(WEATHER_SERVER_PORT),
        ],
        # DEVNULL avoids deadlock when uvicorn fills an unread PIPE buffer.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(WEATHER_SERVER_PORT, timeout=10.0):
        proc.terminate()
        proc.wait()
        pytest.fail(
            f"weather_mcp_server.py did not start on port {WEATHER_SERVER_PORT} within 10 s"
        )
    yield proc
    proc.terminate()
    proc.wait()


@pytest.mark.integration
class TestMCPQuickstartE2E:
    """End-to-end tests for the mcp_dapr_workflow quickstart."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        self.quickstart_dir = quickstarts_dir
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_list_tools_succeeds_and_returns_expected_tools(
        self,
        dapr_runtime,
        weather_mcp_server,  # noqa: ARG002
    ):
        """Quickstart must discover both weather tools from the sidecar."""
        result = run_quickstart_or_examples_script(
            self.quickstart_dir / "mcp_dapr_workflow.py",
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
            use_dapr=True,
            app_id="mcp-agent-test",
            resources_path=self.quickstart_dir / "resources",
        )

        combined = (result.stdout or "") + (result.stderr or "")
        assert "Loaded 2 tool(s)" in combined, (
            f"Expected 'Loaded 2 tool(s)' in output.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "GetWeather" in combined or "get_weather" in combined, (
            "Expected 'GetWeather' tool in output"
        )
        assert "GetForecast" in combined or "get_forecast" in combined, (
            "Expected 'GetForecast' tool in output"
        )

    def test_quickstart_exits_cleanly(
        self,
        dapr_runtime,
        weather_mcp_server,  # noqa: ARG002
    ):
        """Quickstart must exit with return code 0."""
        result = run_quickstart_or_examples_script(
            self.quickstart_dir / "mcp_dapr_workflow.py",
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=120,
            use_dapr=True,
            app_id="mcp-agent-test-exit",
            resources_path=self.quickstart_dir / "resources",
        )

        assert result.returncode == 0, (
            f"Quickstart exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_agent_calls_mcp_tool(
        self,
        dapr_runtime,
        weather_mcp_server,  # noqa: ARG002
    ):
        """The agent must invoke at least one weather tool and return a response."""
        result = run_quickstart_or_examples_script(
            self.quickstart_dir / "mcp_dapr_workflow.py",
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="mcp-agent-tool-call",
            resources_path=self.quickstart_dir / "resources",
        )

        combined = (result.stdout or "") + (result.stderr or "")
        # The agent should produce some weather-related output for Seattle
        assert any(
            kw in combined.lower()
            for kw in ("seattle", "weather", "°f", "sunny", "cloudy", "rainy")
        ), (
            "Expected a weather response for Seattle in output.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


@pytest.mark.integration
class TestMCPIntegrationFailures:
    """Integration failure scenarios for MCP + Dapr workflow."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        self.quickstart_dir = quickstarts_dir
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_list_tools_fails_when_mcp_server_unreachable(self, dapr_runtime):  # noqa: ARG002
        """ListTools workflow must fail (non-zero exit) when the MCP server is down.

        This test intentionally does NOT start weather_mcp_server, so the sidecar
        cannot reach localhost:8081 and the ListTools workflow should fail.
        """
        # Confirm nothing is listening on the weather port before running
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", WEATHER_SERVER_PORT), timeout=0.3)

        result = run_quickstart_or_examples_script(
            self.quickstart_dir / "mcp_dapr_workflow.py",
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=60,
            use_dapr=True,
            app_id="mcp-agent-fail-unreachable",
            resources_path=self.quickstart_dir / "resources",
        )

        combined = (result.stdout or "") + (result.stderr or "")
        assert result.returncode != 0, (
            "Expected non-zero exit when MCP server is unreachable"
        )
        assert any(
            kw in combined
            for kw in ("connection refused", "FAILED", "connect: connection refused")
        ), f"Expected connection-refused error in output.\nCombined:\n{combined}"

    def test_list_tools_fails_when_mcpserver_resource_missing(self, dapr_runtime):  # noqa: ARG002
        """ListTools must fail when no MCPServer resource is loaded by the sidecar."""
        import tempfile

        # Use an empty resources directory (no weather-mcp.yaml)
        with tempfile.TemporaryDirectory() as empty_resources:
            # Copy only the non-MCPServer component YAMLs so Dapr itself starts
            import shutil

            resources = Path(self.quickstart_dir) / "resources"
            for yaml_file in resources.glob("*.yaml"):
                if yaml_file.name != "weather-mcp.yaml":
                    shutil.copy(yaml_file, empty_resources)

            result = run_quickstart_or_examples_script(
                self.quickstart_dir / "mcp_dapr_workflow.py",
                cwd=self.quickstart_dir,
                env=self.env,
                timeout=60,
                use_dapr=True,
                app_id="mcp-agent-fail-no-resource",
                resources_path=Path(empty_resources),
            )

        assert result.returncode != 0, (
            "Expected non-zero exit when MCPServer resource is missing"
        )


@pytest.fixture(scope="module")
def weather_mcp_server_2(quickstarts_dir):
    """Start the second weather MCP server (humidity + wind) on WEATHER_SERVER_PORT_2."""
    proc = subprocess.Popen(
        [
            sys.executable,
            str(quickstarts_dir / "weather_mcp_server_2.py"),
            "--port",
            str(WEATHER_SERVER_PORT_2),
        ],
        # DEVNULL avoids deadlock when uvicorn fills an unread PIPE buffer.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(WEATHER_SERVER_PORT_2, timeout=10.0):
        proc.terminate()
        proc.wait()
        pytest.fail(
            f"weather_mcp_server_2.py did not start on port {WEATHER_SERVER_PORT_2} within 10 s"
        )
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def multi_server_resources_dir(quickstarts_dir):
    """Temp resources dir containing both MCPServer YAMLs + all standard agent YAMLs."""
    base = Path(quickstarts_dir) / "resources"
    tmp = Path(tempfile.mkdtemp(prefix="mcp_multi_resources_"))
    try:
        shutil.copytree(base, tmp, dirs_exist_ok=True)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.integration
class TestMCPServerConfigVariants:
    """Tests for different MCPServer resource configurations."""

    @pytest.fixture(autouse=True)
    def setup(self, quickstarts_dir, openai_api_key):
        self.quickstart_dir = quickstarts_dir
        self.env = {"OPENAI_API_KEY": openai_api_key}

    def test_multiple_servers_both_tool_sets_discovered(
        self,
        dapr_runtime,  # noqa: ARG002
        weather_mcp_server,  # noqa: ARG002
        weather_mcp_server_2,  # noqa: ARG002
        multi_server_resources_dir,
    ):
        """Agent must discover tools from both MCPServer resources (4 tools total).

        weather-mcp.yaml   → port 8081 → get_weather + get_forecast
        weather-mcp-2.yaml → port 8082 → get_humidity + get_wind

        Distinct tool names per server so that auto-discovery does not have to
        dedup; all four tools should register cleanly.
        """
        result = run_quickstart_or_examples_script(
            self.quickstart_dir / "mcp_dapr_workflow_multi.py",
            cwd=self.quickstart_dir,
            env=self.env,
            timeout=180,
            use_dapr=True,
            app_id="mcp-agent-multi-server",
            resources_path=multi_server_resources_dir,
        )

        combined = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, (
            f"Multi-server quickstart exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "Loaded 4 tool(s)" in combined, (
            f"Expected 'Loaded 4 tool(s)' in output.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Both server names must appear
        assert "[weather]" in combined, "Expected '[weather]' server listing in output"
        assert "[weather2]" in combined, (
            "Expected '[weather2]' server listing in output"
        )

    def test_allowed_tools_filter_limits_discovered_tools(
        self,
        dapr_runtime,  # noqa: ARG002
        weather_mcp_server,  # noqa: ARG002
    ):
        """With allowed_tools={"get_weather"}, only 1 tool should be loaded.

        This test runs a modified version of the quickstart that passes
        allowed_tools to the python-sdk DaprMCPClient.  The tool count in
        stdout must reflect the filter.
        """
        import textwrap

        # Inline script: identical to mcp_dapr_workflow.py but with allowed_tools set.
        script_content = textwrap.dedent("""\
            import asyncio, logging, sys
            from dapr.ext.workflow.aio import DaprMCPClient
            logging.basicConfig(level=logging.INFO)

            async def main():
                client = DaprMCPClient(
                    timeout_in_seconds=30,
                    allowed_tools={"get_weather"},
                )
                await client.connect("weather")
                tools = client.get_all_tools()
                print(f"Loaded {len(tools)} tool(s) from MCPServer 'weather':")
                for t in tools:
                    print(f"  * {t.name}: {t.description}")

            asyncio.run(main())
        """)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_allowed_tools_test.py", delete=False
        ) as tmp:
            tmp.write(script_content)
            script_path = Path(tmp.name)
        try:
            result = run_quickstart_or_examples_script(
                script_path,
                cwd=self.quickstart_dir,
                env=self.env,
                timeout=60,
                use_dapr=True,
                app_id="mcp-agent-allowed-tools",
                resources_path=self.quickstart_dir / "resources",
            )
        finally:
            script_path.unlink(missing_ok=True)

        combined = (result.stdout or "") + (result.stderr or "")
        assert result.returncode == 0, (
            f"Allowed-tools quickstart exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "Loaded 1 tool(s)" in combined, (
            f"Expected 'Loaded 1 tool(s)' with allowed_tools filter.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert "GetWeather" in combined or "get_weather" in combined, (
            "Expected only GetWeather tool in output"
        )
