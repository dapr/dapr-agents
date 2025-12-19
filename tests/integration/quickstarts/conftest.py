import os
import signal
import subprocess
import logging
import tempfile
import shutil
import time
import socket
import threading
import re
from pathlib import Path
from typing import Optional, Dict, Any
import pytest

logger = logging.getLogger(__name__)


def setup_quickstart_venv(quickstart_dir: Path, project_root: Path) -> Path:
    """
    Setup a virtual environment for a quickstart so as to better match end user setup.

    Creates a venv in the quickstart directory and installs from requirements.txt.
    So, if we are testing locally then we can update the requirements.txt to use the editable install from the project root.
    We also use uv instead of just pip, as uv is much faster than vanilla pip.

    Args:
        quickstart_dir: Path to the quickstart directory
        project_root: Path to the project root (for editable install if needed)

    Returns:
        Path to the venv Python executable
    """
    # Each quickstart has its own directory, so venv is already unique per quickstart
    # Since pytest-xdist runs one test file per worker, and each quickstart has
    # a unique directory, then we know that each venv path is unique and will not conflict.
    # The venv name is set to ephemeral_test_venv to make it clear these are temporary test venvs.
    venv_path = quickstart_dir / "ephemeral_test_venv"
    if not venv_path.exists():
        logger.info(f"Creating venv in {quickstart_dir}")
        result = subprocess.run(
            ["python3", "-m", "venv", str(venv_path)],
            cwd=quickstart_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create venv: {result.stderr}\n{result.stdout}"
            )

    venv_python = venv_path / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_path / "Scripts" / "python.exe"
    if not venv_python.exists():
        raise RuntimeError(f"Venv Python not found at {venv_python}")

    # Get absolute path without resolving symlinks
    # NOTE: resolving symlinks would point to system Python, but we want venv's Python executable if available
    # The venv's Python is typically a symlink, but we want to use it directly, not resolve it
    venv_python = venv_python.absolute()

    # Also find venv_pip for fallback when uv fails
    venv_pip = venv_path / "bin" / "pip"
    if not venv_pip.exists():
        venv_pip = venv_path / "Scripts" / "pip.exe"

    requirements_file = quickstart_dir / "requirements.txt"
    # Skip installation if already done (for parallel execution)
    installed_marker = venv_path / ".installed"
    if installed_marker.exists():
        logger.info(f"Dependencies already installed for {quickstart_dir}")
    else:
        # Set up environment to ensure uv uses the venv Python
        # Add venv's bin directory to PATH so uv can find the venv Python
        venv_bin = venv_path / "bin"
        if not venv_bin.exists():
            venv_bin = venv_path / "Scripts"

        # Create environment with venv in PATH
        install_env = os.environ.copy()
        venv_bin_str = str(venv_bin.resolve())
        if "PATH" in install_env:
            # Prepend venv bin to PATH so venv Python is found first
            install_env["PATH"] = f"{venv_bin_str}:{install_env['PATH']}"
        else:
            install_env["PATH"] = venv_bin_str

        # Also set VIRTUAL_ENV to help uv detect the venv
        install_env["VIRTUAL_ENV"] = str(venv_path.resolve())

        # Try using uv pip install, with fallback to venv pip if needed
        # Fall back to venv pip if all fail
        def try_uv_install(cmd_args, description):
            """Try uv pip install with different strategies, fall back to venv pip if needed."""
            # First, try uv pip install without --python (relies on PATH/VIRTUAL_ENV)
            result = subprocess.run(
                ["uv", "pip", "install"] + cmd_args,
                cwd=quickstart_dir,
                env=install_env,
                capture_output=True,
                text=True,
                timeout=180,
            )

            # If that works, we're done
            if result.returncode == 0:
                return result

            # Next, try uv pip install with --python flag
            if "externally managed" in result.stderr:
                logger.debug(
                    f"uv pip install without --python failed, trying with --python flag for {description}"
                )
                result = subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(venv_python),
                    ]
                    + cmd_args,
                    cwd=quickstart_dir,
                    env=install_env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

            # If uv still fails with externally managed error, fall back to venv pip
            if result.returncode != 0 and "externally managed" in result.stderr:
                logger.warning(
                    f"uv pip install failed with externally managed error, falling back to venv pip for {description}"
                )
                venv_pip = venv_bin / "pip"
                if not venv_pip.exists():
                    venv_pip = venv_bin / "pip.exe"

                result = subprocess.run(
                    [str(venv_pip)] + ["install"] + cmd_args,
                    cwd=quickstart_dir,
                    env=install_env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install {description}: {result.stderr}\n{result.stdout}"
                )
            return result

        if requirements_file.exists():
            # Install dependencies from requirements.txt first using uv
            logger.info(f"Installing dependencies from {requirements_file} using uv")
            try_uv_install(["-r", str(requirements_file)], "requirements")

            # Override with editable dapr-agents from current repo changes (for PR testing)
            # This ensures we test against the current repo changes, so we test local changes before release
            logger.info(
                f"Installing editable dapr-agents from {project_root} to override requirements.txt"
            )
            try_uv_install(["-e", str(project_root)], "editable dapr-agents")
        else:
            # No requirements.txt - install editable dapr-agents for testing using uv
            logger.info(
                "No requirements.txt found, installing editable dapr-agents using uv"
            )
            try_uv_install(["-e", str(project_root)], "dapr-agents")

        # Mark as installed so if we're running in parallel, we don't reinstall the dependencies.
        installed_marker.touch()

    return venv_python


def _get_project_root(quickstart_dir: Path) -> Path:
    """Get the project root directory from a quickstart directory."""
    return (
        quickstart_dir.parent.parent
        if "quickstarts" in str(quickstart_dir)
        else quickstart_dir.parent
    )


def _setup_venv_and_python(
    quickstart_dir: Path,
    project_root: Path,
    create_venv: bool = True,
) -> tuple[Optional[Path], str]:
    """
    Setup venv and determine Python command to use.

    Returns:
        Tuple of (venv_python_path, python_cmd)
    """
    use_existing_venv = os.getenv("USE_EXISTING_VENV", "").lower() in ("true", "1")
    should_create_venv = create_venv and not use_existing_venv

    venv_python = None
    python_cmd = "python"  # Default fallback

    if should_create_venv:
        venv_python = setup_quickstart_venv(quickstart_dir, project_root)
        if venv_python.exists():
            python_cmd = str(venv_python)
    else:
        if use_existing_venv:
            logger.info(
                f"Using existing venv/system Python for {quickstart_dir} (USE_EXISTING_VENV=true)"
            )
            # When using existing venv, prefer the active venv's Python if available
            # This helps avoid uv warnings about venv mismatches when running with `uv run pytest`
            if "VIRTUAL_ENV" in os.environ:
                venv_python_path = Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"
                if not venv_python_path.exists():
                    venv_python_path = (
                        Path(os.environ["VIRTUAL_ENV"]) / "Scripts" / "python.exe"
                    )
                if venv_python_path.exists():
                    python_cmd = str(venv_python_path)
                    venv_python = venv_python_path
                    logger.debug(f"Using active venv Python: {python_cmd}")
                else:
                    logger.debug(
                        f"VIRTUAL_ENV set to {os.environ['VIRTUAL_ENV']} but Python not found, using system Python"
                    )
        else:
            logger.info(
                f"Using existing venv/system Python for {quickstart_dir} (create_venv=False)"
            )
        # If python_cmd is still "python", it will use system Python or whatever is in PATH

    return venv_python, python_cmd


def _resolve_component_env_vars(
    resources_path: Path,
    cwd_path: Path,
    venv_python: Optional[Path],
    env: dict,
) -> Path:
    """
    Resolve environment variables in component files if needed.

    Returns:
        Path to resolved components directory (may be a temp directory)
    """
    project_root_path = _get_project_root(cwd_path)
    resolve_script = project_root_path / "quickstarts" / "resolve_env_templates.py"

    if resolve_script.exists() and resources_path.exists():
        resolve_python = (
            str(venv_python) if venv_python and venv_python.exists() else "python"
        )
        resolve_result = subprocess.run(
            [resolve_python, str(resolve_script), str(resources_path)],
            cwd=cwd_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if resolve_result.returncode == 0:
            resolved_path = resolve_result.stdout.strip()
            if resolved_path and Path(resolved_path).exists():
                return Path(resolved_path)

    return resources_path


def run_quickstart_script(
    script_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: int = 60,
    use_dapr: bool = False,
    app_id: Optional[str] = None,
    resources_path: Optional[Path] = None,
    dapr_http_port: int = 3500,
    app_port: Optional[int] = None,
    trigger_curl: Optional[Dict[str, Any]] = None,
    trigger_pubsub: Optional[Dict[str, Any]] = None,
    create_venv: bool = True,
    stream_logs: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a quickstart script and return the result.

    This function mimics the user's setup process:
    1. Creates a venv in the quickstart directory (like users do)
    2. Installs from requirements.txt (but uses local editable install for testing)
    3. Runs the script from that venv

    If USE_EXISTING_VENV=true then we will use the system Python venv (current venv or system) instead of creating a new venv.

    Args:
        script_path: Path to the Python script to run
        cwd: Working directory (defaults to script's parent directory)
        env: Environment variables to set
        timeout: Timeout in seconds
        use_dapr: Whether to run with `dapr run`
        app_id: Dapr app ID (required if use_dapr=True)
        resources_path: Path to Dapr resources/components (defaults to cwd/components)
        dapr_http_port: Dapr HTTP port (defaults to 3500)
        app_port: App port for the application (e.g., 8001 for serve mode)
        trigger_curl: Optional dict with curl trigger details. Format:
            {
                "url": "http://localhost:8001/run",
                "method": "POST",
                "data": {"task": "..."},
                "headers": {"Content-Type": "application/json"},
                "wait_seconds": 5  # Time to wait for server to start (default: 5)
            }
        trigger_pubsub: Optional dict with pubsub trigger details. Format:
            {
                "pubsub_name": "messagepubsub",
                "topic": "travel.requests",
                "data": {"task": "..."},
                "wait_seconds": 5  # Time to wait for subscriber to start (default: 5)
            }
        create_venv: Whether to create and use an ephemeral test venv (defaults to True).
                  Set to False or USE_EXISTING_VENV=true to use system Python instead.
        stream_logs: If True, stream stdout/stderr to logger in real-time (defaults to True).
                  Useful for debugging long-running tests.
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    cwd_path = cwd or script_path.parent
    quickstart_dir = cwd_path
    project_root = _get_project_root(quickstart_dir)

    venv_python, python_cmd = _setup_venv_and_python(
        quickstart_dir, project_root, create_venv
    )
    if use_dapr:
        if not app_id:
            raise ValueError("app_id is required when use_dapr=True")
        if not resources_path:
            resources_path = cwd_path / "components"

        resources_path = _resolve_component_env_vars(
            resources_path, cwd_path, venv_python, full_env
        )

        # Build dapr run command
        # Ensure script_path is absolute and exists
        if not script_path.exists():
            raise RuntimeError(f"Script path does not exist: {script_path}")
        script_path_abs = script_path.resolve()

        # Ensure python_cmd is absolute and executable when using venv
        if venv_python and venv_python.exists():
            # Verify the Python executable exists and is executable
            if not os.access(str(venv_python), os.X_OK):
                raise RuntimeError(
                    f"Python executable is not executable: {venv_python}"
                )
            # Use absolute path for Python command
            python_cmd = str(venv_python.absolute())

        cmd = [
            "dapr",
            "run",
            "--app-id",
            app_id,
            "--dapr-http-port",
            str(dapr_http_port),
            "--resources-path",
            str(resources_path),
        ]
        if app_port:
            cmd.extend(["--app-port", str(app_port)])
        cmd.extend(["--", python_cmd, str(script_path_abs)])
    else:
        # Use venv python if available, otherwise system python
        cmd = [python_cmd, str(script_path)]

    # If trigger_curl is provided, we need to run the process in the background,
    # wait for the server to be ready, send the curl request, then terminate
    if trigger_curl:
        return _run_with_curl_trigger(
            cmd, cwd_path, full_env, timeout, trigger_curl, app_port
        )

    # If trigger_pubsub is provided, we need to run the process in the background,
    # wait for the subscriber to be ready, publish a message, then terminate
    if trigger_pubsub:
        return _run_with_pubsub_trigger(
            cmd, cwd_path, full_env, timeout, trigger_pubsub, dapr_http_port
        )

    logger.info(f"running quickstart test cmd {cmd}")
    if stream_logs:
        result = _run_with_streaming(cmd, cwd_path, full_env, timeout)
    else:
        result = subprocess.run(
            cmd,
            cwd=cwd_path,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script_path} failed with return code {result.returncode}.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    return result


def _cleanup_quickstart_venv(quickstart_dir: Path):
    """Helper function to cleanup a single quickstart venv."""
    venv_path = quickstart_dir / "ephemeral_test_venv"
    if venv_path.exists():
        logger.info(f"Removing ephemeral test venv: {venv_path}")
        try:
            shutil.rmtree(venv_path)
        except OSError as e:
            logger.warning(f"Failed to remove {venv_path}: {e}")


@pytest.fixture(scope="module", autouse=True)
def cleanup_quickstart_venv_per_module(request):
    """
    Cleanup ephemeral test venv after all tests in a test module (per quickstart directory) complete.

    This helps free up disk space during test runs, especially for running in CI.
    Each test file typically corresponds to one quickstart or quickstart directory,
    so we clean up that venv after the module's tests complete.
    """
    yield

    # Skip cleanup if using existing venv (for local dev)
    if os.getenv("USE_EXISTING_VENV", "").lower() in ("true", "1"):
        return

    # Try to determine the quickstart directory from the test file path
    # Test files are named like test_01_dapr_agents_fundamentals.py and correspond to quickstarts/01-dapr-agents-fundamentals
    test_file_path = None
    if hasattr(request, "path"):
        test_file_path = Path(request.path)
    elif hasattr(request, "fspath"):
        test_file_path = Path(request.fspath)
    elif hasattr(request, "node") and hasattr(request.node, "fspath"):
        test_file_path = Path(request.node.fspath)

    if test_file_path:
        # Extract quickstart name from test file (e.g., test_01_dapr_agents_fundamentals.py -> 01-dapr-agents-fundamentals)
        test_file_name = test_file_path.stem  # e.g., "test_01_dapr_agents_fundamentals"
        if test_file_name.startswith("test_"):
            # Try to match quickstart directory patterns
            project_root = Path(__file__).parent.parent.parent.parent
            quickstarts_dir = project_root / "quickstarts"

            if quickstarts_dir.exists():
                # Look for matching quickstart directory
                # Test files use underscores, quickstart dirs use hyphens
                quickstart_pattern = test_file_name.replace("test_", "").replace(
                    "_", "-"
                )

                # Try exact match first
                quickstart_dir = quickstarts_dir / quickstart_pattern
                if quickstart_dir.exists():
                    _cleanup_quickstart_venv(quickstart_dir)
                else:
                    # Try to find by number prefix (e.g., "01" -> "01-dapr-agents-fundamentals")
                    test_num = (
                        test_file_name.split("_")[1] if "_" in test_file_name else None
                    )
                    if test_num:
                        for qs_dir in quickstarts_dir.iterdir():
                            if qs_dir.is_dir() and qs_dir.name.startswith(
                                test_num + "-"
                            ):
                                _cleanup_quickstart_venv(qs_dir)
                                break


def run_quickstart_multi_app(
    dapr_yaml_path: Path,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    timeout: int = 180,
    trigger_curl: Optional[Dict[str, Any]] = None,
    trigger_pubsub: Optional[Dict[str, Any]] = None,
    create_venv: bool = True,
    stream_logs: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a quickstart using Dapr multi-app run (`dapr run -f`).

    This function runs multiple apps together using a Dapr multi-app YAML file.
    It mimics the user's setup process by creating venvs and installing dependencies.

    Args:
        dapr_yaml_path: Path to the Dapr multi-app YAML file (e.g., dapr-random.yaml)
        cwd: Working directory (defaults to YAML file's parent directory)
        env: Environment variables to set
        timeout: Timeout in seconds
        trigger_curl: Optional dict with curl trigger details (see run_quickstart_script)
        trigger_pubsub: Optional dict with pubsub trigger details (see run_quickstart_script)
        create_venv: Whether to create and use an ephemeral test venv (defaults to True).
        stream_logs: If True, stream stdout/stderr to logger in real-time (defaults to False).
                  Useful for debugging long-running tests.

    Returns:
        subprocess.CompletedProcess with the result
    """
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    cwd_path = cwd or dapr_yaml_path.parent
    quickstart_dir = cwd_path
    project_root = _get_project_root(quickstart_dir)

    venv_python, _ = _setup_venv_and_python(quickstart_dir, project_root, create_venv)

    # Resolve environment variables in components if needed
    resources_path = quickstart_dir / "components"
    if resources_path.exists():
        resources_path = _resolve_component_env_vars(
            resources_path, cwd_path, venv_python, full_env
        )

    # Build dapr run -f command
    cmd = ["dapr", "run", "-f", str(dapr_yaml_path)]

    # If trigger_curl is provided, we need to run the process in the background,
    # wait for the server to be ready, send the curl request, then terminate
    if trigger_curl:
        # Extract app_port from trigger_curl if available
        app_port = trigger_curl.get("app_port")
        if not app_port and trigger_curl.get("url"):
            try:
                from urllib.parse import urlparse

                parsed = urlparse(trigger_curl["url"])
                if parsed.port:
                    app_port = parsed.port
            except Exception:
                pass
        return _run_with_curl_trigger(
            cmd, cwd_path, full_env, timeout, trigger_curl, app_port
        )

    # If trigger_pubsub is provided, we need to run the process in the background,
    # wait for the subscriber to be ready, publish a message, then terminate
    if trigger_pubsub:
        # Extract dapr_http_port from YAML or use default
        dapr_http_port = 3500  # Default, multi-app run uses default ports
        return _run_with_pubsub_trigger(
            cmd, cwd_path, full_env, timeout, trigger_pubsub, dapr_http_port
        )

    # Run the multi-app command with completion detection
    # dapr run -f runs continuously, and runner.serve() is a long-running service.
    # We need to detect workflow completion and then gracefully shut down.
    if stream_logs:
        result = _run_multi_app_with_completion_detection(
            cmd, cwd_path, full_env, timeout
        )
    else:
        # For non-streaming, we still need completion detection
        # but we'll collect output first, then check for completion
        result = _run_multi_app_with_completion_detection(
            cmd, cwd_path, full_env, timeout
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Multi-app run failed with return code {result.returncode}.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    return result


@pytest.fixture(scope="session", autouse=True)
def cleanup_quickstart_venvs(request):
    """
    Cleanup ephemeral test venvs after all tests complete.

    Note: Venvs are created in each quickstart directory as `ephemeral_test_venv`.
    Example: `quickstarts/01-dapr-agents-fundamentals/ephemeral_test_venv`
    These are cleaned up after tests complete, unless USE_EXISTING_VENV is set.

    This is a fallback cleanup in case the per-module cleanup didn't catch everything after all quickstart tests run.
    """
    yield

    # Skip cleanup if using existing venv (for local dev)
    if os.getenv("USE_EXISTING_VENV", "").lower() in ("true", "1"):
        logger.info("Skipping ephemeral test venv cleanup (USE_EXISTING_VENV=true)")
        return

    # Cleanup ephemeral test venvs
    # Get quickstarts directory from project root (parent conftest fixture path)
    project_root = Path(__file__).parent.parent.parent.parent
    quickstarts_dir = project_root / "quickstarts"

    if quickstarts_dir.exists():
        logger.info("Cleaning up ephemeral test venvs...")
        for quickstart_dir in quickstarts_dir.iterdir():
            if quickstart_dir.is_dir():
                _cleanup_quickstart_venv(quickstart_dir)


def _wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a port to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for a port to become available"""
    return _wait_for_port(host, port, timeout)


class MCPServerContext:
    """
    Context manager for starting and stopping an MCP server in tests.

    Usage:
        with MCPServerContext(quickstart_dir, server_type="sse", port=8000, env={}) as server:
            # Run your test here
            result = run_quickstart_script(...)
    """

    def __init__(
        self,
        quickstart_dir: Path,
        server_type: str = "sse",
        port: int = 8000,
        env: Optional[dict] = None,
        timeout: int = 10,
    ):
        """
        Initialize MCP server context.

        Args:
            quickstart_dir: Directory containing the server.py file
            server_type: Type of server ("sse" or "streamable-http")
            port: Port to run the server on
            env: Environment variables to pass to the server
            timeout: Timeout in seconds to wait for server to start
        """
        self.quickstart_dir = quickstart_dir
        self.server_type = server_type
        self.port = port
        self.env = env or {}
        self.timeout = timeout
        self.server_process = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Start the MCP server and wait for it to be ready."""
        # Setup venv for the quickstart
        project_root = self.quickstart_dir.parent.parent
        venv_python = setup_quickstart_venv(self.quickstart_dir, project_root)
        python_cmd = (
            str(venv_python) if venv_python and venv_python.exists() else "python"
        )

        # Start MCP server in background
        server_script = self.quickstart_dir / "server.py"
        server_cmd = [
            python_cmd,
            str(server_script),
            "--server_type",
            self.server_type,
            "--port",
            str(self.port),
        ]

        self.logger.info(f"Starting MCP server: {' '.join(server_cmd)}")
        full_env = os.environ.copy()
        full_env.update(self.env)

        # Create process in a new process group so we can kill all children
        if os.name == "posix":
            self.server_process = subprocess.Popen(
                server_cmd,
                cwd=self.quickstart_dir,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid,  # Create new process group
            )
        else:  # Windows
            self.server_process = subprocess.Popen(
                server_cmd,
                cwd=self.quickstart_dir,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

        # Wait for MCP server to be ready
        self.logger.info(f"Waiting for MCP server to start on port {self.port}...")
        if not _wait_for_port("localhost", self.port, timeout=self.timeout):
            self._terminate_server()
            raise RuntimeError(f"MCP server failed to start on port {self.port}")

        self.logger.info("MCP server is ready")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminate the MCP server."""
        self._terminate_server()
        return False  # Don't suppress exceptions

    def _terminate_server(self):
        """Terminate the MCP server process."""
        if self.server_process is None:
            return

        self.logger.info("Terminating MCP server...")
        try:
            if os.name == "posix":
                try:
                    pgid = os.getpgid(self.server_process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
            else:
                self.server_process.terminate()
            self.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                "MCP server didn't terminate gracefully, sending SIGKILL..."
            )
            try:
                if os.name == "posix":
                    try:
                        pgid = os.getpgid(self.server_process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    self.server_process.kill()
                self.server_process.wait(timeout=2)
            except Exception as e:
                self.logger.warning(f"Error killing MCP server: {e}")


def _stream_output(pipe, log_func, prefix=""):
    """Stream output from a pipe to a logging function."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                log_func(f"{prefix}{line.rstrip()}")
        pipe.close()
    except Exception as e:
        logger.warning(f"Error streaming output: {e}")


def _run_multi_app_with_completion_detection(
    cmd: list,
    cwd: Path,
    env: dict,
    timeout: int,
    orchestrator_workflow_name: Optional[str] = None,
    grace_period: int = 3,
) -> subprocess.CompletedProcess:
    """
    Run a multi-app command, detect workflow completion, and gracefully shut down.

    This is needed because `dapr run -f` runs continuously and `runner.serve()` is
    a long-running service. We monitor for completion and then send SIGTERM
    to gracefully shut down the orchestrator.

    Args:
        cmd: Command to run
        cwd: Working directory
        env: Environment variables
        timeout: Maximum time to wait
        orchestrator_workflow_name: Expected orchestrator workflow name (e.g., 'random_workflow', 'round_robin_workflow', 'llm_orchestrator_workflow')
        grace_period: Seconds to wait after completion before terminating
    """
    # Infer orchestrator workflow name from dapr YAML filename if not provided
    if orchestrator_workflow_name is None:
        # Try to infer from command (look for dapr-*.yaml)
        for arg in cmd:
            if "dapr-" in arg and arg.endswith(".yaml"):
                yaml_name = Path(arg).stem
                if "random" in yaml_name:
                    orchestrator_workflow_name = "random_workflow"
                elif "roundrobin" in yaml_name or "round-robin" in yaml_name:
                    orchestrator_workflow_name = "round_robin_workflow"
                elif "llm" in yaml_name:
                    orchestrator_workflow_name = "llm_orchestrator_workflow"
                break

    logger.info(f"Running multi-app command with completion detection: {' '.join(cmd)}")
    if orchestrator_workflow_name:
        logger.info(
            f"Looking for orchestrator workflow completion: {orchestrator_workflow_name}"
        )

    # Create process in a new process group so we can kill all children
    # This is important because `dapr run -f` spawns multiple child processes
    # and we need to ensure they all terminate when the test completes
    if os.name == "posix":  # Unix-like systems
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            preexec_fn=os.setsid,  # Create new process group
        )
    else:  # Windows
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows process group
        )

    # Collect output while streaming
    stdout_lines = []
    stderr_lines = []
    orchestrator_completion_detected = False
    orchestrator_failed = False
    completion_time = None
    completed_workflows = set()  # Track unique workflow types
    completion_count = {}  # Track count of each workflow type

    # Pattern to match the specific log line format:
    # "workflow completed with status 'ORCHESTRATION_STATUS_COMPLETED' workflowName '<name>'"
    # Also match FAILED status to detect quickstart failures early
    # This is the only format we want to match - it's from the Workflow Actor log
    completion_pattern = re.compile(
        r"workflow completed with status\s+['\"](ORCHESTRATION_STATUS_COMPLETED|ORCHESTRATION_STATUS_FAILED)['\"]\s+workflowName\s+['\"]([^'\"]+)['\"]",
        re.IGNORECASE,
    )

    def _stream_and_detect(pipe, log_func, prefix, lines_list):
        """Stream output from a pipe and detect completion."""
        nonlocal orchestrator_completion_detected, orchestrator_failed, completion_time, completed_workflows, completion_count  # noqa: F824
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    stripped = line.rstrip()
                    log_func(f"{prefix}{stripped}")
                    lines_list.append(line)

                    # Check for ORCHESTRATION_STATUS_COMPLETED or FAILED messages
                    # Only match the exact format: "workflow completed with status 'ORCHESTRATION_STATUS_*' workflowName '<name>'"
                    match = completion_pattern.search(line)
                    if match:
                        status = match.group(1)
                        workflow_name = match.group(2)
                        if workflow_name:
                            is_failed = "FAILED" in status.upper()
                            completed_workflows.add(workflow_name)
                            completion_count[workflow_name] = (
                                completion_count.get(workflow_name, 0) + 1
                            )
                            total_completions = sum(completion_count.values())
                            # Extract a snippet of the matched line for verification
                            matched_snippet = (
                                match.group(0)[:100]
                                if len(match.group(0)) > 100
                                else match.group(0)
                            )
                            logger.debug(
                                f"Matched line snippet: ...{matched_snippet}... "
                                f"Extracted workflow: {workflow_name}, status: {status}"
                            )
                            status_label = "FAILED" if is_failed else "completed"
                            logger.info(
                                f"Detected workflow {status_label}: {workflow_name} "
                                f"(instance {completion_count[workflow_name]} of this type, "
                                f"{total_completions} total completions)"
                            )

                            # Check if this is the orchestrator's main workflow
                            if (
                                orchestrator_workflow_name
                                and workflow_name == orchestrator_workflow_name
                            ):
                                if not orchestrator_completion_detected:
                                    orchestrator_completion_detected = True
                                    orchestrator_failed = is_failed
                                    completion_time = time.time()
                                    if is_failed:
                                        logger.error(
                                            f"Orchestrator workflow '{orchestrator_workflow_name}' FAILED! "
                                            f"This indicates a quickstart bug, not a test issue."
                                        )
                                    else:
                                        logger.info(
                                            f"Orchestrator workflow '{orchestrator_workflow_name}' completed! "
                                            f"Unique workflow types: {len(completed_workflows)}, "
                                            f"Total completion instances: {total_completions}"
                                        )

                    # Also check for [agent-runner] completion message as a fallback
                    if (
                        not orchestrator_completion_detected
                        and "[agent-runner]" in line
                        and "completed" in line.lower()
                    ):
                        # This is a fallback - if we see agent-runner completion, assume orchestrator is done
                        if (
                            orchestrator_workflow_name is None
                        ):  # Only use fallback if we don't know the workflow name
                            orchestrator_completion_detected = True
                            completion_time = time.time()
                            logger.info("Detected agent-runner completion (fallback)")
            pipe.close()
        except Exception as e:
            logger.warning(f"Error streaming output: {e}")

    # Start threads to stream stdout and stderr
    stdout_thread = threading.Thread(
        target=_stream_and_detect,
        args=(process.stdout, logger.info, "[STDOUT] ", stdout_lines),
    )
    stderr_thread = threading.Thread(
        target=_stream_and_detect,
        args=(process.stderr, logger.warning, "[STDERR] ", stderr_lines),
    )
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    start_time = time.time()
    check_interval = 0.5  # Check every 500ms

    # Wait for orchestrator completion or timeout
    while time.time() - start_time < timeout:
        if orchestrator_completion_detected:
            # Wait for grace period after completion
            elapsed = time.time() - completion_time
            if elapsed >= grace_period:
                # Build summary of completions
                completion_summary = ", ".join(
                    f"{name}({completion_count.get(name, 0)})"
                    for name in sorted(completed_workflows)
                )
                total_completions = sum(completion_count.values())
                logger.info(
                    f"Grace period ({grace_period}s) elapsed after orchestrator completion. "
                    f"Unique workflow types: {len(completed_workflows)} ({completion_summary}), "
                    f"Total completion instances: {total_completions}. "
                    f"Sending SIGTERM to shut down gracefully..."
                )
                # Send SIGTERM to gracefully shut down the entire process group
                try:
                    if os.name == "posix":
                        # Kill the entire process group on Unix
                        try:
                            pgid = os.getpgid(process.pid)
                            os.killpg(pgid, signal.SIGTERM)
                        except (ProcessLookupError, OSError) as e:
                            # Process may have already exited
                            logger.debug(
                                f"Process already exited when sending SIGTERM: {e}"
                            )
                    else:
                        # On Windows, terminate the process (process group handling is different)
                        process.terminate()
                except Exception as e:
                    logger.warning(f"Error sending SIGTERM: {e}")
                break

        time.sleep(check_interval)

        # Check if process already exited
        if process.poll() is not None:
            logger.info("Process exited on its own")
            break

    # Wait for process to terminate gracefully
    if process.poll() is None:
        try:
            process.wait(timeout=10)
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate gracefully, sending SIGKILL...")
            try:
                if os.name == "posix":
                    # Kill the entire process group on Unix
                    try:
                        pgid = os.getpgid(process.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError) as e:
                        # Process may have already exited
                        logger.debug(
                            f"Process already exited when sending SIGKILL: {e}"
                        )
                else:
                    # On Windows, kill the process
                    process.kill()
            except Exception as e:
                # Process may have already exited
                logger.debug(f"Error killing process group: {e}")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process still didn't terminate after SIGKILL")

    # Wait for threads to finish reading
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)

    # Get any remaining output
    try:
        remaining_stdout, remaining_stderr = process.communicate(timeout=1)
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)
    except subprocess.TimeoutExpired:
        pass

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    returncode = process.returncode or 0  # If terminated gracefully, return 0

    # Validate completion
    total_completions = sum(completion_count.values())
    if orchestrator_workflow_name:
        if orchestrator_workflow_name not in completed_workflows:
            completion_summary = ", ".join(
                f"{name}({completion_count.get(name, 0)})"
                for name in sorted(completed_workflows)
            )
            logger.warning(
                f"Orchestrator workflow '{orchestrator_workflow_name}' not found in completed workflows. "
                f"Completed workflows: {completion_summary} (total instances: {total_completions})"
            )
        else:
            completion_summary = ", ".join(
                f"{name}({completion_count.get(name, 0)})"
                for name in sorted(completed_workflows)
            )
            if orchestrator_failed:
                logger.error(
                    f"Orchestrator workflow '{orchestrator_workflow_name}' FAILED. "
                    f"Unique workflow types: {len(completed_workflows)} ({completion_summary}), "
                    f"Total completion instances: {total_completions}"
                )
            else:
                logger.info(
                    f"Successfully detected orchestrator workflow '{orchestrator_workflow_name}' completion. "
                    f"Unique workflow types: {len(completed_workflows)} ({completion_summary}), "
                    f"Total completion instances: {total_completions}"
                )

    # If orchestrator failed, raise an error immediately
    if orchestrator_failed:
        raise RuntimeError(
            f"Orchestrator workflow '{orchestrator_workflow_name}' failed. "
            f"Check the logs above for details."
        )

    if not orchestrator_completion_detected and time.time() - start_time >= timeout:
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)

    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _run_with_streaming(
    cmd: list,
    cwd: Path,
    env: dict,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run a command with streaming output to logs."""
    logger.info(f"Running command with streaming logs: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Collect output while streaming
    stdout_lines = []
    stderr_lines = []

    def _stream_and_collect(pipe, log_func, prefix, lines_list):
        """Stream output from a pipe to a logging function and collect it."""
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    stripped = line.rstrip()
                    log_func(f"{prefix}{stripped}")
                    lines_list.append(line)
            pipe.close()
        except Exception as e:
            logger.warning(f"Error streaming output: {e}")

    # Start threads to stream stdout and stderr
    stdout_thread = threading.Thread(
        target=_stream_and_collect,
        args=(process.stdout, logger.info, "[STDOUT] ", stdout_lines),
    )
    stderr_thread = threading.Thread(
        target=_stream_and_collect,
        args=(process.stderr, logger.warning, "[STDERR] ", stderr_lines),
    )
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"Process timed out after {timeout}s, killing...")
        process.kill()
        returncode = process.wait()
        # Wait for threads to finish reading
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        # Get any remaining output
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=1)
            if remaining_stdout:
                stdout_lines.append(remaining_stdout)
            if remaining_stderr:
                stderr_lines.append(remaining_stderr)
        except subprocess.TimeoutExpired:
            pass
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)

    # Wait for threads to finish reading
    stdout_thread.join(timeout=2)
    stderr_thread.join(timeout=2)

    # Get any remaining output
    try:
        remaining_stdout, remaining_stderr = process.communicate(timeout=1)
        if remaining_stdout:
            stdout_lines.append(remaining_stdout)
        if remaining_stderr:
            stderr_lines.append(remaining_stderr)
    except subprocess.TimeoutExpired:
        pass

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _run_with_curl_trigger(
    cmd: list,
    cwd: Path,
    env: dict,
    timeout: int,
    trigger_curl: Dict[str, Any],
    app_port: Optional[int],
) -> subprocess.CompletedProcess:
    """Run a command with a curl trigger for workflows that need to be triggered via curl."""
    import requests

    # Start the process in the background
    logger.info(f"Starting process in background: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for the server to be ready
        wait_seconds = trigger_curl.get("wait_seconds", 5)
        url = trigger_curl.get("url", "")

        # Extract port from URL if app_port not provided
        if app_port is None and url:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if parsed.port:
                    app_port = parsed.port
            except Exception:
                pass

        if app_port:
            logger.info(f"Waiting for server to start on port {app_port}...")
            if not _wait_for_port("localhost", app_port, timeout=wait_seconds + 10):
                logger.warning(
                    f"Port {app_port} not ready after {wait_seconds + 10}s, proceeding anyway"
                )
        else:
            logger.info(f"Waiting {wait_seconds}s for server to start...")
            time.sleep(wait_seconds)

        # Send the curl request
        method = trigger_curl.get("method", "POST")
        data = trigger_curl.get("data", {})
        headers = trigger_curl.get("headers", {"Content-Type": "application/json"})

        logger.info(f"Sending {method} request to {url}")
        try:
            if method.upper() == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            else:
                response = requests.request(
                    method, url, json=data, headers=headers, timeout=30
                )

            logger.info(f"Received response: {response.status_code}")
            curl_output = f"HTTP {response.status_code}\n{response.text}"
        except Exception as e:
            logger.warning(f"Curl request failed: {e}")
            curl_output = f"Curl request failed: {e}"

        # Wait a bit for the process to handle the request
        time.sleep(2)

        # Terminate the process
        logger.info("Terminating process...")
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

        # Combine outputs
        combined_stdout = f"{stdout}\n\n--- Curl Response ---\n{curl_output}"

        return subprocess.CompletedProcess(
            cmd, process.returncode, combined_stdout, stderr
        )

    except Exception as e:
        # Make sure to clean up the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
        raise RuntimeError(f"Error running with curl trigger: {e}")


def _run_with_pubsub_trigger(
    cmd: list,
    cwd: Path,
    env: dict,
    timeout: int,
    trigger_pubsub: Dict[str, Any],
    dapr_http_port: int,
) -> subprocess.CompletedProcess:
    """Run a command with a pubsub trigger for subscriber-based scripts."""
    # Start the subscriber process in the background
    logger.info(f"Starting subscriber process in background: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for the subscriber to be ready
        wait_seconds = trigger_pubsub.get("wait_seconds", 5)
        logger.info(f"Waiting {wait_seconds}s for subscriber to start...")
        time.sleep(wait_seconds)

        pubsub_name = trigger_pubsub.get("pubsub_name", "messagepubsub")
        topic = trigger_pubsub.get("topic", "travel.requests")
        data = trigger_pubsub.get("data", {})

        # Build payload for dapr publish
        import json

        if isinstance(data, dict):
            payload_json = json.dumps(data)
        else:
            payload_json = json.dumps({"task": str(data)})

        publish_cmd = [
            "dapr",
            "publish",
            "--dapr-http-port",
            str(dapr_http_port),
            "--pubsub",
            pubsub_name,
            "--topic",
            topic,
            "--data",
            payload_json,
        ]

        logger.info(f"Publishing message using: {' '.join(publish_cmd)}")
        try:
            client_result = subprocess.run(
                publish_cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if client_result.returncode == 0:
                logger.info("Published message successfully via dapr publish")
                pubsub_output = (
                    f"Published via dapr publish\nSTDOUT: {client_result.stdout}\n"
                    f"STDERR: {client_result.stderr}"
                )
            else:
                logger.warning(
                    f"dapr publish returned non-zero: {client_result.returncode}"
                )
                pubsub_output = (
                    f"dapr publish failed (code {client_result.returncode})\n"
                    f"STDOUT: {client_result.stdout}\nSTDERR: {client_result.stderr}"
                )
        except Exception as e:
            logger.warning(f"Pubsub publish failed: {e}")
            pubsub_output = f"Pubsub publish failed: {e}"

        # Wait for the workflow to complete (give it time to process)
        logger.info("Waiting for workflow to process message...")
        time.sleep(10)  # Give the workflow time to process

        logger.info("Terminating subscriber process...")
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()

        # Combine outputs
        combined_stdout = f"{stdout}\n\n--- Pubsub Trigger ---\n{pubsub_output}"

        return subprocess.CompletedProcess(
            cmd, process.returncode, combined_stdout, stderr
        )

    except Exception as e:
        # Clean up the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
        raise RuntimeError(f"Error running with pubsub trigger: {e}")
